library(shiny)
library(shinydashboard)
library(DT)
library(tm)
library(SnowballC)
library(textclean)
library(caret)
library(e1071)
library(glmnet)

# Load and clean dataset
spam_data <- read.csv("C:/Users/Faizan/Desktop/visio/elections/spam.csv", 
                      stringsAsFactors = FALSE,
                      fileEncoding = "UTF-8-BOM")

# Rename and clean
spam_data <- spam_data[, 1:2]
colnames(spam_data) <- c("label", "message")
spam_data$label <- factor(spam_data$label)

# Text cleaning function
clean_corpus <- function(text) {
  text <- replace_non_ascii(text)
  corpus <- VCorpus(VectorSource(text))
  corpus <- tm_map(corpus, content_transformer(tolower))
  corpus <- tm_map(corpus, removePunctuation)
  corpus <- tm_map(corpus, removeNumbers)
  corpus <- tm_map(corpus, removeWords, stopwords("english"))
  corpus <- tm_map(corpus, stemDocument)
  corpus <- tm_map(corpus, stripWhitespace)
  return(corpus)
}

# Preprocess
corpus <- clean_corpus(spam_data$message)
dtm <- DocumentTermMatrix(corpus)
dtm <- removeSparseTerms(dtm, 0.99)

X <- as.data.frame(as.matrix(dtm))
X$label <- spam_data$label

# Split data
set.seed(123)
splitIndex <- createDataPartition(X$label, p = 0.8, list = FALSE)
train_data <- X[splitIndex, ]
test_data <- X[-splitIndex, ]

# Train model
model <- train(label ~ ., data = train_data, method = "glmnet")
predictions <- predict(model, test_data)
conf_matrix <- confusionMatrix(predictions, test_data$label)

# UI
ui <- dashboardPage(
  dashboardHeader(title = "📧 Spam Detection App"),
  dashboardSidebar(
    sidebarMenu(
      menuItem(" Data Preview", tabName = "data_preview", icon = icon("table")),
      menuItem(" Model Evaluation", tabName = "model_eval", icon = icon("chart-bar")),
      menuItem(" Spam Predictor", tabName = "predictor", icon = icon("robot"))
    )
  ),
  dashboardBody(
    tabItems(
      tabItem(tabName = "data_preview",
              fluidRow(
                box(title = "📑 Full Dataset", width = 12, status = "primary", solidHeader = TRUE,
                    DTOutput("data_table"))
              )
      ),
      tabItem(tabName = "model_eval",
              fluidRow(
                box(title = "📉 Confusion Matrix", width = 12, status = "info", solidHeader = TRUE,
                    verbatimTextOutput("conf_matrix"))
              )
      ),
      tabItem(tabName = "predictor",
              fluidRow(
                box(title = "💬 Enter a Message Below", width = 12, status = "success", solidHeader = TRUE,
                    textAreaInput("user_input", "Your message:", "", rows = 6, width = "100%"),
                    actionButton("predict_btn", "🔍 Predict"),
                    tags$hr(),
                    verbatimTextOutput("prediction_result"))
              )
      )
    )
  )
)

# Server
server <- function(input, output, session) {
  
  output$data_table <- renderDT({
    datatable(spam_data, options = list(scrollX = TRUE, pageLength = 10))
  })
  
  output$conf_matrix <- renderPrint({
    conf_matrix
  })
  
  observeEvent(input$predict_btn, {
    req(input$user_input)
    
    input_text <- input$user_input
    input_text <- replace_non_ascii(input_text)
    corpus_new <- clean_corpus(input_text)
    
    dtm_new <- DocumentTermMatrix(corpus_new, control = list(dictionary = Terms(dtm)))
    df_new <- as.data.frame(as.matrix(dtm_new))
    
    # Ensure matching columns
    missing_cols <- setdiff(colnames(train_data)[-ncol(train_data)], colnames(df_new))
    for (col in missing_cols) df_new[, col] <- 0
    df_new <- df_new[, colnames(train_data)[-ncol(train_data)]]
    
    prediction <- predict(model, df_new)
    
    output$prediction_result <- renderPrint({
      paste("🔎 Predicted Label:", as.character(prediction))
    })
  })
}

# Run the app
shinyApp(ui, server)
