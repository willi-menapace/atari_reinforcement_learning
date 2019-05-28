library(ggplot2)

first = read.csv('01.csv')
#second = read.csv('02.csv')
#third = read.csv('03.csv')

#merged = rbind(rbind(first, second), third)

plot = ggplot(first, aes(Step, Value)) +
  geom_line() +
  xlab("Training iteration") +
  ylab("Reward") +
  

print(plot)