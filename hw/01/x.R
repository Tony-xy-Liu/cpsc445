# install.packages('readxl')
library(readxl)
q3_data <- read_excel("/home/tony/Downloads/hw-data.xlsx","q3")
q3_df <- data.frame(y = as.vector( t(q3_data) ),
                    group = rep( c('A','B','C','D'), nrow(q3_data) ))
fit = lm(y~group, data=q3_df)
summary(fit)
anova(fit)



q4_data <- read_excel("/home/tony/Downloads/hw-data.xlsx","q4")
fit_g1 = lm(g1~cond+type, data=q4_data)
fit_g2 = lm(g2~cond+type, data=q4_data)
summary(fit_g1)
summary(fit_g2)


library(ggplot2)
q4_data$res.g1 = fit_g1$residuals
q4_data$res.g2 = fit_g2$residuals
ggplot(q4_data) +
  geom_point(aes(x=cond, y=res.g1), color='red') +
  geom_point(aes(x=type, y=res.g1), color='blue') +
  scale_x_discrete(name = "predictors", labels=c("type: ct1", "type: ct2", "cond: ctrl", "cond: disease"))
ggplot(q4_data) +
  geom_point(aes(x=cond, y=res.g2), color='red') +
  geom_point(aes(x=type, y=res.g2), color='blue') +
  scale_x_discrete(name = "predictors", labels=c("type: ct1", "type: ct2", "cond: ctrl", "cond: disease"))


fit_g1_2 = lm(g1~cond+type+g2, data=q4_data)
summary(fit_g1_2)



q4_data$type = as.factor(q4_data$type)
fit_lg1 = glm(type~g1+cond, data=q4_data, family=binomial)
fit_lg2 = glm(type~g2+cond, data=q4_data, family=binomial)
summary(fit_lg1)
summary(fit_lg2)

1-mean((fit_lg1$fitted.values<0.5) != (q4_data$type == "ct1"))
1-mean((fit_lg2$fitted.values<0.5) != (q4_data$type == "ct1"))



fit_pr = lm(g2~g1, data=q4_data)
sign(fit_pr$coefficients[2]) * sqrt(summary(fit_pr)$r.squared)
cor(q4_data$g1, q4_data$g2, method="pearson")

q4_data$res.g1 = fit_g1$residuals
q4_data$res.g2 = fit_g2$residuals
cor(q4_data$res.g1, q4_data$res.g2, method="pearson")






library(readxl)
q5_data <- read_excel("/home/tony/Downloads/hw-data.xlsx","q5")
fit_pos = glm(cell~cond, data=q5_data, family=poisson(link="log"), offset(log(total)))
summary(fit_pos)

library("MASS")
fit_nb = glm.nb(cell~cond, data=q5_data, offset(log(total)))
summary(fit_nb)

fit_bin = glm(cell~cond, data=q5_data, family=binomial, offset(log(total)))
summary(fit_bin)


q5_data$cell_prob = q5_data$cell/q5_data$total
fit_bin = glm(cell_prob~cond, data=q5_data, family=binomial(link="logit"), offset(log(total)))
summary(fit_bin)




