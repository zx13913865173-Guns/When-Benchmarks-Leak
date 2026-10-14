# Step 6: Mixed-effects models with FDR correction

library(lme4)
library(lmerTest)
library(effectsize)

df <- read.csv("outputs/metrics/per_image_metrics.csv", stringsAsFactors = FALSE)
df_hybrid <- df[df$type %in% c("primary_hybrid", "generalization_hybrid"), ]
df_hybrid$weight_ratio <- paste(df_hybrid$weight_a, df_hybrid$weight_b, sep = ":")
df_hybrid$style_position <- ifelse(df_hybrid$dominant == df_hybrid$style_a,
                                    "dominant", "secondary")

cat("\n=== Model 1: Weight ratio x style position ===\n")
model1 <- lmer(global_recall ~ weight_ratio * style_position +
               (1 | style_a) + (1 | style_b), data = df_hybrid)
print(summary(model1))
print(anova(model1))

p_vals <- summary(model1)$coefficients[, "Pr(>|t|)"]
p_fdr <- p.adjust(p_vals, method = "BH")
cat("\nFDR-adjusted p-values:\n")
print(p_fdr)

cat("\nPartial eta-squared:\n")
print(eta_squared(model1, partial = TRUE))

cat("\n=== Model 2: Regional vs global ===\n")
df_long <- reshape(
  df_hybrid[, c("filename", "global_recall", "regional_recall")],
  direction = "long",
  varying = c("global_recall", "regional_recall"),
  v.names = "recall",
  times = c("global", "regional"),
  timevar = "condition",
  idvar = "filename"
)
model2 <- lmer(recall ~ condition + (1 | filename), data = df_long)
print(summary(model2))
print(anova(model2))

cat("\nStatistical analysis complete.\n")
