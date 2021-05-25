#' @author Andre Bauer

#' @description Automatically evaluates and ranks forecasting methods based on their performance.
#'
#' @title Execute the benchmark
#' @param forecaster The forecasting method. This method gets a timeseries objekt (ts) and the horizon (h). The method returns the forecast values.
#' @param usecase The use case for the benchmark. It must be either economics, finance, human, or nature.
#' @param type Optional parameter: The evaluation type. It must be either one (one-step-ahead forecast), multi (multi-step-ahead forecast), or rolling (rolling-origin forecast). one by default.
#' @param output Optional parameter: The name of the output file. benchmark.csv by default.
#' @param name Optional parameter: The name of the forecasting method. Benchmarked Method by default.
#' @return The performance of the forecasting method in comparison with state-of-the-art methods.
#' @examples
#' # Example usage
#' benchmark(forecaster,usecase="economics",type="one")
#'
#' # Example forecasting method
#' forecaster <- function(ts,h){ return(forecast(ets(ts), h = h)$mean) }
#' @export
benchmark <- function(forecaster, usecase, type = "one", output="benchmark.csv", name="Benchmarked Method"){
  # Gives feedback about input
  print(paste("Selected use case is '", usecase, "'", sep=""))
  print(paste("Selected evaluation type is '", type, "'", sep=""))

  usecase <- tolower(usecase)
  type <- tolower(type)

  # Configures the benchmark
  switch(usecase,
         "economics" = {
            data <- economics
            ind <- 1:100
         },
         "finance" = {
            data <- finance
            ind <- 101:200
         },
         "human" = {
            data <- human_access
            ind <- 201:300
         },
         "nature" = {
            data <- nature_and_demographic
            ind <- 301:400
         }, {
           stop("The wrong use case was selected. It must be either economics, finance, human, or nature.")
         }
  )

  # Performs the benchmarking
  results <- evaluation(forecaster, data, type)

  # Adds performance of state-of-the-art methods
  switch(type,
         "one" = {
           time.mean <- c(mean(results[,1]), colMeans(benchmark.one.time[ind,]))
           smape.mean <- c(mean(results[,2]), colMeans(benchmark.one.smape[ind,]))
           mase.mean <- c(mean(results[,3]), colMeans(benchmark.one.mase[ind,]))
           mues.mean <- c(mean(results[,4]), colMeans(benchmark.one.mues[ind,]))
           moes.mean <- c(mean(results[,5]), colMeans(benchmark.one.moes[ind,]))
           muas.mean <- c(mean(results[,6]), colMeans(benchmark.one.muas[ind,]))
           moas.mean <- c(mean(results[,7]), colMeans(benchmark.one.moas[ind,]))

           time.sd <- c(sd(results[,1]), apply(benchmark.one.time[ind,], 2, sd))
           smape.sd <- c(sd(results[,2]), apply(benchmark.one.smape[ind,], 2, sd))
           mase.sd <- c(sd(results[,3]), apply(benchmark.one.mase[ind,], 2, sd))
           mues.sd <- c(sd(results[,4]), apply(benchmark.one.mues[ind,], 2, sd))
           moes.sd <- c(sd(results[,5]), apply(benchmark.one.moes[ind,], 2, sd))
           muas.sd <- c(sd(results[,6]), apply(benchmark.one.muas[ind,], 2, sd))
           moas.sd <- c(sd(results[,7]), apply(benchmark.one.moas[ind,], 2, sd))
         },
         "multi" = {
           time.mean <- c(mean(results[,1]), colMeans(benchmark.multi.time[ind,]))
           smape.mean <- c(mean(results[,2]), colMeans(benchmark.multi.smape[ind,]))
           mase.mean <- c(mean(results[,3]), colMeans(benchmark.multi.mase[ind,]))
           mues.mean <- c(mean(results[,4]), colMeans(benchmark.multi.mues[ind,]))
           moes.mean <- c(mean(results[,5]), colMeans(benchmark.multi.moes[ind,]))
           muas.mean <- c(mean(results[,6]), colMeans(benchmark.multi.muas[ind,]))
           moas.mean <- c(mean(results[,7]), colMeans(benchmark.multi.moas[ind,]))

           time.sd <- c(sd(results[,1]), apply(benchmark.multi.time[ind,], 2, sd))
           smape.sd <- c(sd(results[,2]), apply(benchmark.multi.smape[ind,], 2, sd))
           mase.sd <- c(sd(results[,3]), apply(benchmark.multi.mase[ind,], 2, sd))
           mues.sd <- c(sd(results[,4]), apply(benchmark.multi.mues[ind,], 2, sd))
           moes.sd <- c(sd(results[,5]), apply(benchmark.multi.moes[ind,], 2, sd))
           muas.sd <- c(sd(results[,6]), apply(benchmark.multi.muas[ind,], 2, sd))
           moas.sd <- c(sd(results[,7]), apply(benchmark.multi.moas[ind,], 2, sd))
         },
         "rolling" = {
           time.mean <- c(mean(results[,1]), colMeans(benchmark.rolling.time[ind,]))
           smape.mean <- c(mean(results[,2]), colMeans(benchmark.rolling.smape[ind,]))
           mase.mean <- c(mean(results[,3]), colMeans(benchmark.rolling.mase[ind,]))
           mues.mean <- c(mean(results[,4]), colMeans(benchmark.rolling.mues[ind,]))
           moes.mean <- c(mean(results[,5]), colMeans(benchmark.rolling.moes[ind,]))
           muas.mean <- c(mean(results[,6]), colMeans(benchmark.rolling.muas[ind,]))
           moas.mean <- c(mean(results[,7]), colMeans(benchmark.rolling.moas[ind,]))

           time.sd <- c(sd(results[,1]), apply(benchmark.rolling.time[ind,], 2, sd))
           smape.sd <- c(sd(results[,2]), apply(benchmark.rolling.smape[ind,], 2, sd))
           mase.sd <- c(sd(results[,3]), apply(benchmark.rolling.mase[ind,], 2, sd))
           mues.sd <- c(sd(results[,4]), apply(benchmark.rolling.mues[ind,], 2, sd))
           moes.sd <- c(sd(results[,5]), apply(benchmark.rolling.moes[ind,], 2, sd))
           muas.sd <- c(sd(results[,6]), apply(benchmark.rolling.muas[ind,], 2, sd))
           moas.sd <- c(sd(results[,7]), apply(benchmark.rolling.moas[ind,], 2, sd))
         }
  )



  methods <- c(name, "ETS", "sARIMA", "sNaive", "TBATS", "Theta", "GPyTorch", "NNetar",
               "Random Forest", "SVR", "XGBoost")

  # Prints the different performance measures
  names(time.mean) <- methods
  print("## Avg. Normalized Time")
  print(time.mean)

  names(time.sd) <- methods
  print("## SD. Normalized Time")
  print(time.sd)

  names(smape.mean) <- methods
  print("## Avg. Symmetrical Mean Absolute Percentage Error")
  print(smape.mean)

  names(smape.sd) <- methods
  print("## SD. Symmetrical Mean Absolute Percentage Error")
  print(smape.sd)

  names(mase.mean) <- methods
  print("## Avg. Mean Absolute Scaled Error")
  print(mase.mean)

  names(mase.sd) <- methods
  print("## SD. Mean Absolute Scaled Error")
  print(mase.sd)

  names(mues.mean) <- methods
  print("## Avg. Mean Under-Estimation Share")
  print(mues.mean)

  names(mues.sd) <- methods
  print("## SD. Mean Under-Estimation Share")
  print(mues.sd)

  names(moes.mean) <- methods
  print("## Avg. Mean Over-Estimation Share")
  print(moes.mean)

  names(moes.sd) <- methods
  print("## SD. Mean Over-Estimation Share")
  print(moes.sd)

  names(muas.mean) <- methods
  print("## Avg. Mean Under-Accuracy Share")
  print(muas.mean)

  names(muas.sd) <- methods
  print("## SD. Mean Under-Accuracy Share")
  print(muas.sd)

  names(moas.mean) <- methods
  print("## Avg. Mean Over-Accuracy Share")
  print(moas.mean)

  names(moas.sd) <- methods
  print("## SD. Mean Over-Accuracy Share")
  print(moas.sd)

  # Prepares the output
  result <- rbind(time.mean, time.sd, smape.mean, smape.sd, mase.mean, mase.sd, mues.mean, mues.sd,
                  moes.mean, moes.sd, muas.mean, muas.sd, moas.mean, moas.sd)
  rownames(result) <- c("Avg. Normalized Time", "SD. Normalized Time",
                        "Avg. Symmetrical Mean Absolute Percentage Error", "SD. Symmetrical Mean Absolute Percentage Error",
                        "Avg. Mean Absolute Scaled Error", "SD. Mean Absolute Scaled Error",
                        "Avg. Mean Under-Estimation Share", "SD. Mean Under-Estimation Share",
                        "Avg. Mean Over-Estimation Share", "SD. Mean Over-Estimation Share",
                        "Avg. Mean Under-Accuracy Share", "SD. Mean Under-Accuracy Share",
                        "Avg. Mean Over-Accuracy Share", "SD. Mean Over-Accuracy Share")

  # Writes benchmarking results
  write.table(result,file=output,sep = ";", col.names=NA)

}

plot_values = function(df, maxval, title, ylabel, xlabel) {
  # https://stackoverflow.com/a/29463136
  library("dplyr")
  library("grid") # needed for arrow() function
  
  df = stack(df)
  p = ggplot(df, aes(x=ind, y=values)) + 
    xlab(xlabel) + 
    ylab(ylabel) + 
    ggtitle(title) + 
    theme(plot.title = element_text(hjust = 0.5))
  # p + geom_boxplot()
  
  
  dd <- df %>% filter(values>maxval) %>%
    group_by(ind) %>%
    summarise(outlier_txt=paste(values, collapse=", "))
  # p
  
  
  p2 <- p + geom_boxplot() +
    scale_y_continuous(limits=c(min(df$values),maxval))+
    geom_text(data=dd,aes(y=maxval,label=outlier_txt),
              size=3,vjust=-3.5,hjust=0.5)+
    geom_segment(data=dd,aes(y=maxval*0.95,yend=maxval,
                             xend=ind),
                 arrow = arrow(length = unit(0.3,"cm")))
  p2
}

plot_metadata = function(type){
  library(ggplot2)
  
  datasets = list(economics, finance, human_access, nature_and_demographic)
  c_total_lengths = c()
  c_hist_lengths = c()
  c_horizons = c()
  for(d in datasets) {
    total_lengths = c()
    hist_lengths = c()
    horizons = c()
    for(i in 1:100){
      total_lengths = c(total_lengths, length(d[[i]]))
      hist_length = (ceiling(length(d[[i]]) * 0.8))
      hist_lengths = c(hist_lengths, hist_length)
      horizons = c(horizons, length(d[[i]]) - hist_length)
    }
    c_total_lengths = c(c_total_lengths, list(total_lengths))
    c_hist_lengths = c(c_hist_lengths, list(hist_lengths))
    c_horizons = c(c_horizons, list(horizons))
  }
  usecase_names = c("Economics", "Finance", "Human", "Nature")
  df_total_lengths = setNames(data.frame(c_total_lengths), usecase_names)
  df_hist_lengths = setNames(data.frame(c_hist_lengths), usecase_names)
  df_horizons = setNames(data.frame(c_horizons), usecase_names)
  
  # ggplot(stack(df_horizons), aes(x=ind, y=values)) + geom_violin()
  # plot_values(df_total_lengths, 15000)
  plot_values(df_hist_lengths, 13000, "History Length vs Use Case", "History Length", "Use Case")
  # plot_values(df_horizons, 2500)
}
plot_metadata("one")
