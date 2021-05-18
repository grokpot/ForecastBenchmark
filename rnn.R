library(ForecastBenchmark)
# https://www.google.com/url?sa=t&rct=j&q=&esrc=s&source=web&cd=&ved=2ahUKEwjI2fOM6NLwAhWRGewKHRr0BCwQFjAEegQIEBAD&url=https%3A%2F%2Fpiazza.com%2Fclass_profile%2Fget_resource%2Fjqsdh4th7et3gg%2Fjtry7ie0dh6373&usg=AOvVaw2PbPuuxhVwXXJg_poU6rnv
library(keras)
# library(pROC)

rnn <- function(ts, h) {
    # imdb <- dataset_imdb(num_words = 10)

    ep <- length(ts) * 0.8
    print(train_data)

    c(train_data, train_labels) %<-% imdb$train
    c(test_data, test_labels) %<-% imdb$test

    print(train_data)

    # maxlen <- 256
    # train_data <- pad_sequences(train_data, maxlen = maxlen)
    # test_data <- pad_sequences(test_data, maxlen = maxlen)

    stop

    # Specify network architecture
    model1 <- keras_model_sequential() %>%
        layer_embedding(input_dim = 10000, output_dim = 32) %>%
        layer_simple_rnn(units = 32) %>%
        layer_dense(units = 1, activation = "sigmoid")

    # Specify Loss and Optimizer
    model1 %>% compile(
        optimizer = "rmsprop",
        loss = "binary_crossentropy",
        metrics = c("acc")
    )
    summary(model1)

    # Model Fitting and Evaluation
    history <- model1 %>% fit(
        train_data, train_labels,
        epochs = 10,
        batch_size = 128
    )

    plot(history)

    stop
}

benchmark(rnn, usecase = "human", type = "rolling", output = "/Users/prater/Dev/fhnw-aci-libra/results/nature_rolling.csv")