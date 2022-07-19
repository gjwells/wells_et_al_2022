################################################################################.
#### Models for Wells et al. 2022 ##############################################.
################################################################################.
# See Wells et al. 2022, 'Nature entwined: 700 million people need wild harvests 
# and economic development for wellbeing'

# N.B. The script is designed to run several models in parallel on 
# ~20 to 40 cores. If running with fewer cores, it will likely a take a long
# time (e.g. 4 to 5 days for the whole script on only 4 cores)

#### BACKGROUND ################################################################

# This script runs a series of logistic Bayesian GLMMs to test associations
# between household and settlement-level attributes and three different outcomes: the
# probability of a household wild harvesting, and the probability of a wild harvesting
# household being above a wellbeing threshold for food security and life satisfaction.

# Several different models are run for each outcome to test against the null model, 
# and for spatial LOO cross-validation.

# To aid with computation, we first set up all of the models we want to run,
# then run these in parallel (using mclapply).

# The structure of the script is as follows:
# - Load data
# - Check for multicollinearity of predictors
# - Set up a list with inputs and arguments for each model
# - Run main models in parallel

#### SETUP #####################################################################
script_start <- Sys.time()

### . packages ----
library(Hmisc)
library(readr)
library(dplyr)
library(tidyr)
library(lme4)
library(usdm)
library(rstanarm)
library(bayestestR)
library(bayesplot)
library(parallel)
library(stringi)

### . set max default cores to use ----
#options(mc.cores = parallel::detectCores())
#options(mc.cores = 20)
options(mc.cores = 2)

### . set random seed ----
SEED=14124869

### . save directory ----

# save object name and path. set to where you want list of models (can be large)
sav_obj <- paste("C:/test/",Sys.Date(),".rda",sep="")

# empty list for results object
results <- list()

#### LOAD AND PREPARE DATA #####################################################

### . a) load csv from p1_data_gen.R script ----
dat <- read.csv(encoding="UTF-8",stringsAsFactors = FALSE, 
                  "./data/dat.csv")

### . b) scale some variables to help with model convergence ----
# make dataframe with scaling factors and keep to rescale for plotting later
scl_info <- data.frame(
  pop_dens = 7000,
  nt_lght = 54
)
results$scl_info <- scl_info

# loop to scale variables
for (i in 1:ncol(scl_info)){
  dat[names(scl_info)[i]] <- 
    dat[names(scl_info)[i]]/scl_info[,names(scl_info)[i]][1]
}

### . c) set data types ----
# make selected variables ordered factors, make gen_hd factor
dat <- dat %>%
  mutate_at(vars(-pop_dens:-lat,-gen_hd,-lvlhd_total_q),~ordered(.)) %>%
  mutate_at(vars(gen_hd),~factor(.))

#### MULTICOLLINEARITY CHECKS ##################################################

### . a) check vif and correlations of predictors ----
# vif
tmp <- dat %>%
  dplyr::select(uncult_hv_bn:nt_lght) %>%
  mutate_all(as.numeric) %>%
  as.data.frame()
usdm::vif(tmp)
# correlation matrix
round(Hmisc::rcorr(as.matrix(tmp))$r,2)

### . b) remove problematic variables ----

# remove problematic vars from main dataframe
# population density highly correlated with night lights
dat$pop_dens <- NULL
# check again
# vif
tmp <- dat %>%
  dplyr::select(uncult_hv_bn:nt_lght) %>%
  mutate_all(as.numeric) %>%
  as.data.frame()
usdm::vif(tmp)
round(Hmisc::rcorr(as.matrix(tmp))$r,2)
# all ok

#### SET UP AND RUN ALL MODEL RUNS #############################################
# - All models are Bayesian logistic hierarchical models (GLMMs) with random intercepts

### Model structure
# - see the paper for the reasoning behind the model structures
# - For fixed effects, I use two different structures depending on the outcome:
#   - for outcome 'presence of wild harvesting': all HH- and village-level predictors without interactions
#   - for wellbeing outcomes: HH- and village-level predictors with interactions
#     with presence of wild harvesting (to see how association between WHV and 
#     WB changes with HH and settlement atrributes)
# - For random intercepts, I use :
#   - settlement (sett_id): the village with which a household is associated.
#   - OECD Development Assistance Committee dac_reg (dac_reg): the global region
#     with which a region is associated (Sub-Saharan Africa, South Asia, 
#     Latin America & Caribbean, East Asia & Pacific )
# - For each outcome I test the following random effects structures:
#   - null model (settlement random intercept only)
#   - random intercepts for sett_id
#   - random intercepts for sett_id and dac_reg
 
### Priors
# - using weakly informative prior with logit link, as per Gelman et al. 2008
# http://www.stat.columbia.edu/~gelman/research/published/priors11.pdf
# - prior has df 7, mean 0, scale 2.5 which according to Gelman et al. 2008 for logistic regression
# 'outperforms the normal, on average, because it allows for occasional large coefficients while
# still performing a reasonable amount of shrinkage for coefficients near zero'.
# i.e. the distribution allows for longer tails

### Code
# - I use stan_glmer from rstanarm to run the models
# - the code works by making a list of all model calls, then sending these to
# different cores using mclapply
# - this list includes the calls for our actual models, and for the spatial
# leave one out cross-validation (SLOO) models (see paper for details)
# - I first set up the generic model structures and a function to simplify 
# populating stan_glmer. I then send the list of model calls to this function.

### . a) set up model structures ----
# to save on code I make generic model structures, then later gsub in the 
# outcome variables and ranef structures for each model within the function

## . . i) wild harvest outcome ----

# wild harvest null
whv_null <- "outcome ~ 1 + (1|sett_id)"

# wild harvest ranefs
whv_int <- "outcome ~ cult_hv_bn + other_inc_bn + lvlhd_total_q + 
                   edu_yrs + ls_pas + gen_hd + tn_ext_reg + 
                   city_dist + nt_lght + natlc_pc +
                   ranefs"

## . . ii) wellbeing outcomes ----

# wellbeing null
wb_null <- "outcome ~ 1 + (1|sett_id)"

# wellbeing ranefs
wb_int <- "outcome ~
                    uncult_hv_bn*cult_hv_bn +
                    uncult_hv_bn*other_inc_bn +
                    uncult_hv_bn*lvlhd_total_q +
                    uncult_hv_bn*edu_yrs +
                    uncult_hv_bn*ls_pas +
                    uncult_hv_bn*gen_hd +
                    uncult_hv_bn*tn_ext_reg +
                    uncult_hv_bn*city_dist +
                    uncult_hv_bn*nt_lght +
                    uncult_hv_bn*natlc_pc +
                    ranefs"

### . b) make function to populate stan_glmer ----
# allows for standard and sloo model runs

## . . i) global function settings ----

# stan_glmer set number of cores to use.
# only use 1 in stan_glmer when using mclapply. otherwise mclapply throws errors (and you end up using ALOT of computing power)
n_cores = 1 

# stan_glmer set number of chains. usualy 4, lower for testing
n_chains = 4
#n_chains = 1

# stan_glmer. usually 4000. make lower for testing
iterations = 4000
#iterations = 100

# stan_glmer prior
t_prior = student_t(df = 7, location = 0.0, scale = 2.5)

# sloo: number of test points
sloo.ntest = 50

# sloo: sample of testing points
# systematic random sample (grid)
sloo.smp <- round(seq(100,
                      by=(nrow(dat[!is.na(dat$lat),])-1)/sloo.ntest,
                      length.out=sloo.ntest
                      ))

## . . ii) function ----
func_stan_glmer <- function(dpvr,fm,indat,is.sloo,sloo.index,sloo.buffer){
  # dpvr = name of outcome variable
  # fm = formula (model structure)
  # indat = input data frame
  # is.sloo = TRUE/FALSE, if the model is part of spatial loo validation
  # sloo.index = for sloo, number of index from random sample. ignored if is.sloo is false
  # sloo.buffer = for sloo, distance of buffer in dec deg
  
  # out list
  out <- list()
  
  # set up formula
  fm_in <- gsub("outcome",dpvr,fm) # add outcome name
  fm_in <- gsub("\n","",fm_in) # remove new line syntax
  
  ### Run standard GLMM if not SLOO validation
  if(is.sloo==F){
    # make model name: outcome plus model structure
    mdnm <- paste("md_",dpvr,"_",sep="") # dep var
    
    if(grepl("\\(uncult",fm_in)==F){mdnm <-paste(mdnm,"it_",sep="")} # intercepts/slopes/null
    if(grepl("~ 1 +",fm_in)==T){mdnm <-gsub("it_","null_",mdnm)}
    
    if(grepl("sett_id\\)",fm_in)){mdnm <-paste(mdnm,"st",sep="")} # ranefs
    if(grepl("cn\\)",fm_in)){mdnm <-paste(mdnm,"cn",sep="")}
    if(grepl("dac_reg\\)",fm_in)){mdnm <-paste(mdnm,"dc",sep="")}
    
    start_time <- Sys.time()
    message("********","\n",
            paste("Starting model:",mdnm),"\n",
            paste("Start time:",start_time),"\n",
            "********")
    
    md <- stan_glmer(fm_in, data = indat,
                     family = binomial(link = "logit"),
                     prior = t_prior, prior_intercept = t_prior,
                     seed = SEED, iter = iterations,
                     cores = n_cores, chains = n_chains,
                     adapt_delta = 0.99
    )
    
    # keep and name results. name = dep var + model type
    out$tp <- md
    names(out)[length(out)] <- mdnm
    
    message("********","\n",
            paste("Completed model:",mdnm),"\n",
            paste("Finish time:",Sys.time()),"\n")
    message(paste("Total runtime:",round((Sys.time()-start_time)/60,1), " minutes"))
    message("********")
    
    out
    
  }
  
  ### If SLOO validation
  # SLOO validation compares predictions with spatially proximate observations
  # removed against prediction with full dataset. If similar, then little spatial 
  # autocorrelation.
  
  if(is.sloo==T){
    # make model name: outcome plus model structure
    mdnm <- paste("loo_",dpvr,"_",sep="") # dep var
    
    if(grepl("\\(uncult",fm_in)==F){mdnm <-paste(mdnm,"it_",sep="")} # intercepts/slopes/null
    if(grepl("\\(uncult",fm_in)==T){mdnm <-paste(mdnm,"itsl_",sep="")}
    if(grepl("~ 1 +",fm_in)==T){mdnm <-gsub("it_","null_",mdnm)}
    
    if(grepl("sett_id\\)",fm_in)){mdnm <-paste(mdnm,"st",sep="")} # ranefs
    if(grepl("cn\\)",fm_in)){mdnm <-paste(mdnm,"cn",sep="")}
    if(grepl("dac_reg\\)",fm_in)){mdnm <-paste(mdnm,"dc",sep="")}
    start_time <- Sys.time()
    message("********","\n",
            paste("Starting an S-LOO test for:",mdnm),"\n",
            paste("Test no.:",sloo.index),"\n",
            paste("Start time:",start_time),"\n",
            "********")
    
    # get data with spatial locations
    mddat <- indat %>% filter(!is.na(lat))
    
    ## run SLOO (removing spatially proximate obs)
    # get test point
    test <- mddat[sloo.smp[sloo.index],]
    
    # get training data (with test point & obs. within buffer removed)
    train <- mddat[sqrt((mddat[,"lon"]-test[,"lon"])^2 +
                          (mddat[,"lat"]-test[,"lat"])^2)>sloo.buffer,]
    
    # build the sloo model with training data
    md <- stan_glmer(fm_in, data = train,
                     family = binomial(link = "logit"),
                     prior = t_prior, prior_intercept = t_prior,
                     seed = SEED, iter = iterations,
                     cores = n_cores, chains = n_chains)
    
    # predict on test point
    p <- posterior_predict(md, newdata=test)
    
    # get results as data.frame
    res <- data.frame(
      spatial.loo = T,
      buffer_dd = sloo.buffer,
      model = mdnm,
      obs_index = test$hhid,
      observed = test[,dpvr],
      predicted = median(p) # get median posterior prediction
    )
    
    ## run standard LOO (with spatially proximate obs)
    # first run only
    # get test point
    test <- mddat[sloo.smp[sloo.index],]
    
    # get training data (with test point & obs. within buffer removed)
    train <- mddat[mddat$hhid!=test$hhid,]
    
    # build the model with training data
    md <- stan_glmer(fm_in, data = train,
                     family = binomial(link = "logit"),
                     prior = t_prior, prior_intercept = t_prior,
                     seed = SEED, iter = iterations,
                     cores = n_cores, chains = n_chains)
    
    # predict on test point
    p <- posterior_predict(md, newdata=test)
    
    # get results as data.frame
    res <- bind_rows(res,data.frame(
      spatial.loo = F,
      buffer_dd = sloo.buffer,
      model = mdnm,
      obs_index = test$hhid,
      observed = test[,dpvr],
      predicted = median(p) # get median posterior prediction
    ) )
    
    # keep results
    if(length(out)>0){out[[length(out)+1]] <- res}
    if(length(out)==0){out[[1]] <- list(res)}
    names(out)[length(out)] <- mdnm
    
    
    message("********","\n",
            paste("Completed model:",mdnm),"\n",
            paste("Finish time:",Sys.time()),"\n")
    message("Total runtime:")
    message(Sys.time()-start_time)
    message("********")
    
    out
    
  }
  
  out
  
}

### . c) make list of calls for mclapply ----
# Two type of calls:
# - standard calls
# - calls for spatial loo testing

# Make calls to populate: func_stan_glmer <- function(dpvr,fm,indat,is.sloo,sloo.index,sloo.buffer)
# dpvr = name of outcome variable
# fm = formula
# indat = input data frame
# is.sloo = TRUE/FALSE, if the model is part of spatial loo validation
# sloo.index = for sloo, number of index from random sample. ignored if is.sloo is false
# sloo.buffer = for sloo, distance of buffer in dec deg

## . . i) null models ----

calls <- list(
  ### wild harvesting
  # null
  list(dpvr = "uncult_hv_bn",
       fm = whv_null,
       indat = dat,
       is.sloo = F,
       sloo.index = NA,
       sloo.buffer = NA),
  
  ### food security
  # null
  list(dpvr = "hlt_fd_sec",
       fm = whv_null,
       indat = dat,
       is.sloo = F,
       sloo.index = NA,
       sloo.buffer = NA),
  
  ### life satisfaction
  # null
  list(dpvr = "swb",
       fm = whv_null,
       indat = dat,
       is.sloo = F,
       sloo.index = NA,
       sloo.buffer = NA)
  
  
)

## . . ii) random intercepts ----
# generate in loops, gsub in ranefs

### ranef intercept structturs
ref <- c(
  "(1|sett_id)",
  "(1|sett_id) + (1|dac_reg)"
)

### wild harvesting
for(i in 1:length(ref)){
  calls[[length(calls)+1]] <- list(dpvr = "uncult_hv_bn", 
                                   fm = gsub("ranefs",ref[i], whv_int),
                                   indat = dat,
                                   is.sloo = F,
                                   sloo.index = NA,
                                   sloo.buffer = NA)
}

# food security
for(i in 1:length(ref)){
  calls[[length(calls)+1]] <- list(dpvr = "hlt_fd_sec", 
                                   fm = gsub("ranefs",ref[i], wb_int),
                                   indat = dat,
                                   is.sloo = F,
                                   sloo.index = NA,
                                   sloo.buffer = NA)
}

# life satsifaction
for(i in 1:length(ref)){
  calls[[length(calls)+1]] <- list(dpvr = "swb", 
                                   fm = gsub("ranefs",ref[i], wb_int),
                                   indat = dat,
                                   is.sloo = F,
                                   sloo.index = NA,
                                   sloo.buffer = NA)
}

# view model structures in calls
tmp <- data.frame()
for(i in 1:length(calls)){
  tmp <- bind_rows(tmp,
    do.call(bind_rows,calls[[i]][-3])
    )
}
tmp$fm <- gsub("\n","",tmp$fm) # print last 50 chars of fm to help visualise
tmp$fm <- gsub(" ","",tmp$fm)
tmp$fm <- stri_sub(tmp$fm,-50)
nmn <- nrow(tmp)
message("Main model structures (fm = last 50 chars of formula)")
print(tmp[c(1,3:5,2)])


# ## . . iv) sloo runs for selected wild harvesting model ----
# selected from above main models using LOO-IC after initial run
# generate in for loop to get number of runs needed, with different test points
# and buffers

# double loop to generate for for different buffers
bf <- c(1,3,5,7,9) # buffer in dd

for(k in bf){
  for(i in 1:sloo.ntest){
    tp <- list(dpvr = "uncult_hv_bn",
               fm = gsub("ranefs","(1|sett_id) + (1|dac_reg)", whv_int),
               indat = dat,
               is.sloo = T,
               sloo.index = i,
               sloo.buffer = k)
    calls[[length(calls)+1]] <- tp
  }
}

# view model structures in calls
tmp <- data.frame()
for(i in 1:length(calls)){
  tmp <- bind_rows(tmp,
                   do.call(bind_rows,calls[[i]][-3])
  )
}
tmp$fm <- gsub("\n","",tmp$fm) # print last 50 chars of fm to help visualise
tmp$fm <- gsub(" ","",tmp$fm)
tmp$fm <- stri_sub(tmp$fm,-50)
tmp <- tmp %>% mutate_at(vars(dpvr, fm , is.sloo, sloo.buffer),~factor(.))
message("Summary of SLOO model structures.  (fm = last 50 chars of formula)")
print(summary(tmp[-1:-nmn, c(1,3:5,2)]))


### . d) run models ----

message(paste("CALLS EXIST? ",exists("calls"),sep="") )

## . . i) run mclapply ----
res <- parallel::mclapply(X=calls,
                               FUN=function(x)do.call(func_stan_glmer,x))

## . . ii) get results in correct formats ----

# extract and standards glmms and loo results
res <- unlist(res,recursive=F)
# models
mds <- res[grepl("md_",names(res))]
# loos
loos <- res[grepl("loo_",names(res))]
loos <- do.call(bind_rows, loos)

# add to results
results <- c(results,mds)
results$loo_results <- loos

## .  save results ----
save(results, file = sav_obj)
