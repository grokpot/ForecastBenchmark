library(ForecastBenchmark)

# forecaster <- function(ts, h) {
#     values <- thetaf(ts, h = h)$mean
#     return(values)
# }

forecaster <- function(ts, h) {
    # print(paste(c("The first three notes are: ", ts[1]), collapse = " "))
    print(frequency(ts))
    return(forecast(ets(ts), h = h)$mean)
}

# benchmark(forecaster,usecase="nature",type="multi",output="/results/nature_multi.csv")
# benchmark(forecaster, usecase = "human", type = "rolling", output = "/Users/prater/Dev/fhnw-aci-libra/results/nature_rolling.csv")
benchmark(forecaster, usecase = "finance", type = "one", output = "/results/finance_one.csv")
# benchmark(forecaster,usecase="economics",type="rolling",output="/results/economics_rolling.csv")