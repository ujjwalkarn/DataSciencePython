library(jsonlite)
options(encoding = "UTF-8")

# read in individual JSON lines
json_file <- "C:\\Users\\ujjwal.karn\\Desktop\\Tweets\\python.json"

# turn it into a proper array by separating each object with a "," and
# wrapping that up in an array with "[]"'s.

dat <- fromJSON(sprintf("[%s]", paste(readLines(json_file), collapse=",")))

dim(dat)
## [1] 3959   18

tweets<-dat$text
tweets
