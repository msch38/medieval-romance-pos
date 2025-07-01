# ============================================================================
# Medieval Languages POS Tagging - LLM Prompting
# ============================================================================


# Install and load required packages
if (!require("pacman")) install.packages("pacman")
pacman::p_load(
  tidyverse, ggplot2, viridis, patchwork, scales, 
  broom, ggpubr, RColorBrewer, gridExtra, cowplot
)

# Set global theme
theme_set(theme_minimal(base_size = 11) +
            theme(
              plot.title = element_text(size = 13, face = "bold", hjust = 0.5),
              plot.subtitle = element_text(size = 10, hjust = 0.5),
              legend.position = "bottom",
              panel.grid.minor = element_blank()
            ))

# Custom colors
model_colors <- c("GEMMA3" = "#ff6b6b", "PHI4" = "#27ae60")
prompting_colors <- c("zero" = "#ff6b6b", "few" = "#27ae60")
dataset_colors <- c("NAF" = "#ff6b6b", "CAT" = "#f39c12", "CHAULIAC" = "#27ae60")

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

`%||%` <- function(x, y) if (is.null(x) || is.na(x) || length(x) == 0) y else x

# String concatenation
`%.%` <- function(x, y) paste0(x, y)

calculate_cohens_d <- function(x, y) {
  # Ensure x and y are numeric vectors
  x <- as.numeric(x)
  y <- as.numeric(y)
  
  n1 <- length(x)
  n2 <- length(y)
  
  if (n1 < 2 || n2 < 2) {
    warning("Not enough data points for Cohen's d calculation")
    return(NA)
  }
  
  pooled_sd <- sqrt(((n1 - 1) * var(x) + (n2 - 1) * var(y)) / (n1 + n2 - 2))
  (mean(x) - mean(y)) / pooled_sd
}

interpret_effect_size <- function(d) {
  d_abs <- abs(d)
  case_when(
    is.na(d_abs) ~ "unknown",
    d_abs < 0.2 ~ "negligible",
    d_abs < 0.5 ~ "small", 
    d_abs < 0.8 ~ "medium",
    TRUE ~ "large"
  )
}

# ============================================================================
# DATA PARSING FUNCTION
# ============================================================================

parse_reports <- function(file_path = "C:/Users/esteb/Downloads/MELT_analysis/combined_classification_reports.txt") {
  cat("📂 Parsing classification reports from:", file_path, "\n")
  
  if (!file.exists(file_path)) {
    cat("❌ File not found. ")
    return()
  }
  
  tryCatch({
    content <- read_file(file_path)
    blocks <- str_split(content, "===REPORT_START===")[[1]][-1]
    
    results <- map_dfr(blocks, function(block) {
      # Extract fields using regex
      model <- str_extract(block, "(?<=MODEL: )\\w+")
      prompting <- str_extract(block, "(?<=PROMPTING: )\\w+") 
      dataset <- str_extract(block, "(?<=DATASET: )\\w+")
      decoding <- str_extract(block, "(?<=DECODING: )\\w+")
      accuracy <- as.numeric(str_extract(block, "(?<=Accuracy: )[0-9.]+"))
      
      # Return data if essential fields exist
      if (!any(is.na(c(model, prompting, dataset, accuracy)))) {
        tibble(
          model = toupper(str_trim(model)),
          prompting = str_trim(prompting),
          dataset = toupper(str_trim(dataset)),
          decoding = str_trim(decoding %||% "unknown"),
          accuracy = accuracy
        )
      }
    })
    
    if (nrow(results) == 0) {
      cat("❌ No valid data found. ")
      return()
    }
    
    # Convert to factors
    results$model <- factor(results$model, levels = c("GEMMA3", "PHI4"))
    results$prompting <- factor(results$prompting, levels = c("zero", "few"))
    results$dataset <- factor(results$dataset, levels = c("NAF", "CAT", "CHAULIAC"))
    
    cat("✅ Parsed", nrow(results), "experiments successfully\n")
    return(results)
    
  }, error = function(e) {
    cat("❌ Error parsing file:", e$message, "\n")
    return()
  })
}


# ============================================================================
# STATISTICAL ANALYSIS FUNCTIONS
# ============================================================================

perform_statistical_analysis <- function(df) {
  cat("\n", strrep("=", 60), "\n")
  cat("📊 STATISTICAL SIGNIFICANCE TESTING\n")
  cat(strrep("=", 60), "\n")
  
  results <- list()
  
  # === MODEL ANALYSIS ===
  cat("\n🤖 MODEL PERFORMANCE ANALYSIS\n")
  cat(strrep("-", 35), "\n")
  
  model_aov <- aov(accuracy ~ model, data = df)
  model_summary <- summary(model_aov)
  
  # Effect size
  ss_total <- sum((df$accuracy - mean(df$accuracy))^2)
  ss_between <- model_summary[[1]]$`Sum Sq`[1]
  eta_squared <- ss_between / ss_total
  
  cat(sprintf("Model ANOVA: F(%d,%d) = %.2f, p = %.2e\n",
              model_summary[[1]]$Df[1],
              model_summary[[1]]$Df[2], 
              model_summary[[1]]$`F value`[1],
              model_summary[[1]]$`Pr(>F)`[1]))
  cat(sprintf("Effect size (η²) = %.3f\n", eta_squared))
  
  # Pairwise comparisons
  tukey_results <- TukeyHSD(model_aov)
  cat("\nPairwise comparisons:\n")
  for (i in 1:nrow(tukey_results$model)) {
    comp <- rownames(tukey_results$model)[i]
    diff <- tukey_results$model[i, "diff"] 
    p_val <- tukey_results$model[i, "p adj"]
    
    models <- str_split(comp, "-")[[1]]
    group1 <- df$accuracy[df$model == str_trim(models[2])]
    group2 <- df$accuracy[df$model == str_trim(models[1])]
    cohens_d <- calculate_cohens_d(group1, group2)
    
    cat(sprintf("  %s: Δ=%.4f, p=%.2e, d=%.2f (%s)\n", 
                comp, diff, p_val, cohens_d, interpret_effect_size(cohens_d)))
  }
  
  results$model_analysis <- list(anova = model_aov, eta_squared = eta_squared)
  
  # === PROMPTING ANALYSIS ===
  cat("\n💭 PROMPTING STRATEGY ANALYSIS\n")
  cat(strrep("-", 35), "\n")
  
  # Extract vectors instead of data frames
  zero_group <- df$accuracy[df$prompting == "zero"]
  few_group <- df$accuracy[df$prompting == "few"]
  
  # Check if we have data in both groups
  if (length(zero_group) == 0 || length(few_group) == 0) {
    cat("❌ Missing data for one or both prompting groups\n")
    return(results)
  }
  
  prompting_test <- t.test(zero_group, few_group)
  cohens_d_prompting <- calculate_cohens_d(zero_group, few_group)
  
  cat(sprintf("Zero-shot: n=%d, mean=%.4f±%.4f\n", 
              length(zero_group), mean(zero_group), sd(zero_group)))
  cat(sprintf("Few-shot:  n=%d, mean=%.4f±%.4f\n", 
              length(few_group), mean(few_group), sd(few_group)))
  cat(sprintf("t-test: t(%.1f)=%.2f, p=%.3f\n", 
              prompting_test$parameter, prompting_test$statistic, prompting_test$p.value))
  cat(sprintf("Cohen's d = %.2f (%s)\n", 
              cohens_d_prompting, interpret_effect_size(cohens_d_prompting)))
  
  results$prompting_analysis <- list(ttest = prompting_test, cohens_d = cohens_d_prompting)
  
  # === DATASET ANALYSIS ===
  cat("\n📚 DATASET DIFFICULTY ANALYSIS\n")
  cat(strrep("-", 35), "\n")
  
  dataset_aov <- aov(accuracy ~ dataset, data = df)
  dataset_summary <- summary(dataset_aov)
  
  ss_dataset <- dataset_summary[[1]]$`Sum Sq`[1]
  eta_squared_dataset <- ss_dataset / ss_total
  
  cat(sprintf("Dataset ANOVA: F(%d,%d) = %.2f, p = %.3f\n",
              dataset_summary[[1]]$Df[1],
              dataset_summary[[1]]$Df[2],
              dataset_summary[[1]]$`F value`[1], 
              dataset_summary[[1]]$`Pr(>F)`[1]))
  cat(sprintf("Effect size (η²) = %.3f\n", eta_squared_dataset))
  
  dataset_means <- df %>%
    group_by(dataset) %>%
    summarise(n = n(), mean = mean(accuracy), sd = sd(accuracy), .groups = 'drop')
  
  for (i in 1:nrow(dataset_means)) {
    cat(sprintf("  %s: n=%d, mean=%.4f±%.4f\n",
                dataset_means$dataset[i], dataset_means$n[i], 
                dataset_means$mean[i], dataset_means$sd[i]))
  }
  
  results$dataset_analysis <- list(anova = dataset_aov, eta_squared = eta_squared_dataset)
  
  return(results)
}

# ============================================================================
# VISUALIZATION FUNCTIONS
# ============================================================================

create_model_plot <- function(df) {
  # Summary stats for labels
  model_stats <- df %>%
    group_by(model) %>%
    summarise(mean_acc = mean(accuracy), .groups = 'drop')
  
  ggplot(df, aes(x = model, y = accuracy, fill = model)) +
    geom_violin(alpha = 0.7, trim = FALSE) +
    geom_boxplot(width = 0.2, fill = "white", alpha = 0.8) +
    geom_jitter(width = 0.1, alpha = 0.6, size = 1.5) +
    scale_fill_manual(values = model_colors) +
    scale_y_continuous(labels = percent_format(accuracy = 0.1), 
                       limits = c(0.35, 0.9)) +
    labs(title = "Model Performance Distribution",
         subtitle = "Accuracy across all experiments",
         x = "Model", y = "Accuracy") +
    theme(legend.position = "none") +
    geom_text(data = model_stats, 
              aes(x = model, y = 0.37, label = sprintf("μ=%.3f", mean_acc)),
              inherit.aes = FALSE, size = 3, fontface = "bold")
}

create_prompting_plot <- function(df) {
  prompting_stats <- df %>%
    group_by(prompting) %>%
    summarise(mean_acc = mean(accuracy), 
              se = sd(accuracy)/sqrt(n()), .groups = 'drop')
  
  ggplot(prompting_stats, aes(x = prompting, y = mean_acc, fill = prompting)) +
    geom_col(alpha = 0.8, width = 0.6) +
    geom_errorbar(aes(ymin = mean_acc - se, ymax = mean_acc + se),
                  width = 0.2, size = 1) +
    scale_fill_manual(values = prompting_colors,
                      labels = c("Zero-shot", "Few-shot")) +
    scale_y_continuous(labels = percent_format(accuracy = 0.1),
                       limits = c(0.6, 0.8)) +
    labs(title = "Prompting Strategy Comparison", 
         subtitle = "Mean accuracy ± standard error",
         x = "Prompting Strategy", y = "Mean Accuracy") +
    theme(legend.position = "none") +
    geom_text(aes(label = sprintf("%.1f%%", mean_acc*100)), 
              vjust = -1.5, fontface = "bold")
}

create_interaction_plot <- function(df) {
  interaction_data <- df %>%
    group_by(model, prompting) %>%
    summarise(mean_acc = mean(accuracy),
              se = sd(accuracy)/sqrt(n()), .groups = 'drop')
  
  ggplot(interaction_data, aes(x = model, y = mean_acc, fill = prompting)) +
    geom_col(position = position_dodge(0.8), alpha = 0.8) +
    geom_errorbar(aes(ymin = mean_acc - se, ymax = mean_acc + se),
                  position = position_dodge(0.8), width = 0.3) +
    scale_fill_manual(values = prompting_colors,
                      labels = c("Zero-shot", "Few-shot")) +
    scale_y_continuous(labels = percent_format(accuracy = 0.1)) +
    labs(title = "Model × Prompting Interaction",
         subtitle = "Mean accuracy by model and prompting strategy", 
         x = "Model", y = "Mean Accuracy", fill = "Strategy") +
    theme(legend.position = "bottom")
}

create_scatter_plot <- function(df) {
  ggplot(df, aes(x = model, y = accuracy, color = model, shape = prompting)) +
    geom_jitter(width = 0.2, alpha = 0.7, size = 2.5) +
    scale_color_manual(values = model_colors) +
    scale_shape_manual(values = c(16, 17), labels = c("Zero-shot", "Few-shot")) +
    scale_y_continuous(labels = percent_format(accuracy = 0.1),
                       limits = c(0.35, 0.9)) +
    labs(title = "Individual Experiment Results",
         subtitle = "All experiments by model and prompting strategy",
         x = "Model", y = "Accuracy", 
         color = "Model", shape = "Prompting") +
    guides(color = guide_legend(override.aes = list(size = 4)),
           shape = guide_legend(override.aes = list(size = 4)))
}

create_dataset_plot <- function(df) {
  ggplot(df, aes(x = dataset, y = accuracy, fill = dataset)) +
    geom_violin(alpha = 0.7) +
    geom_boxplot(width = 0.3, fill = "white", alpha = 0.8) +
    geom_jitter(width = 0.1, alpha = 0.5, size = 1.5) +
    scale_fill_manual(values = dataset_colors) +
    scale_y_continuous(labels = percent_format(accuracy = 0.1)) +
    scale_x_discrete(labels = c("NAF\n(Medieval\nOccitan)", 
                                "CAT\n(Medieval\nCatalan)", 
                                "CHAULIAC\n(Medieval\nFrench)")) +
    labs(title = "Dataset Difficulty Analysis",
         subtitle = "Performance across medieval language datasets",
         x = "Dataset", y = "Accuracy") +
    theme(legend.position = "none")
}

create_effect_sizes_plot <- function(stats_results) {
  # Extract effect sizes from actual results
  effect_data <- tibble(
    Comparison = c("Model Effects", "Prompting Effect", "Dataset Effect"),
    Effect_Size = c(
      stats_results$model_analysis$eta_squared %||% 0.5,
      stats_results$prompting_analysis$cohens_d %||% 0.1,
      stats_results$dataset_analysis$eta_squared %||% 0.2
    ),
    Type = c("Model", "Prompting", "Dataset")
  )
  
  effect_data$Magnitude <- case_when(
    effect_data$Effect_Size < 0.2 ~ "Negligible",
    effect_data$Effect_Size < 0.5 ~ "Small", 
    effect_data$Effect_Size < 0.8 ~ "Medium",
    TRUE ~ "Large"
  )
  
  effect_data$Magnitude <- factor(effect_data$Magnitude, 
                                  levels = c("Negligible", "Small", "Medium", "Large"))
  
  ggplot(effect_data, aes(x = reorder(Comparison, Effect_Size), 
                          y = Effect_Size, fill = Magnitude)) +
    geom_col(alpha = 0.8) +
    coord_flip() +
    scale_fill_manual(values = c("Negligible" = "#bdc3c7", "Small" = "#f39c12",
                                 "Medium" = "#e67e22", "Large" = "#e74c3c")) +
    geom_hline(yintercept = c(0.2, 0.5, 0.8), linetype = "dashed", alpha = 0.5) +
    labs(title = "Effect Sizes",
         subtitle = "Magnitude of experimental effects",
         x = "Comparison", y = "Effect Size", fill = "Magnitude") +
    theme(legend.position = "bottom")
}

# ============================================================================
# MAIN ANALYSIS EXECUTION
# ============================================================================

run_complete_analysis <- function() {
  cat("🚀 MEDIEVAL LANGUAGES POS TAGGING ANALYSIS\n")
  cat(strrep("=", 50), "\n")
  
  # Load data
  df <- parse_reports()
  
  # Show data summary
  cat("\n📊 DATASET OVERVIEW:\n")
  cat("Total experiments:", nrow(df), "\n")
  cat("Models:", paste(levels(df$model), collapse = ", "), "\n")
  cat("Accuracy range:", sprintf("%.3f - %.3f", min(df$accuracy), max(df$accuracy)), "\n")
  
  # Statistical analysis
  stats_results <- perform_statistical_analysis(df)
  
  # Create visualizations
  cat("\n📈 Creating visualizations...\n")
  
  tryCatch({
    p1 <- create_model_plot(df)
    p2 <- create_prompting_plot(df)  
    p3 <- create_interaction_plot(df)
    p4 <- create_scatter_plot(df)
    p5 <- create_dataset_plot(df)
    p6 <- create_effect_sizes_plot(stats_results)
    
    # Combine plots
    layout <- (p1 | p2 | p5) / (p3 | p4 | p6)
    final_plot <- layout + plot_annotation(
      title = "Medieval Languages POS Tagging: Statistical Analysis Dashboard",
      subtitle = sprintf("Comprehensive analysis of %d experiments", nrow(df)),
      caption = "Comprehensive statistical analysis with effect sizes",
      theme = theme(plot.title = element_text(size = 16, face = "bold", hjust = 0.5),
                    plot.subtitle = element_text(size = 12, hjust = 0.5),
                    plot.caption = element_text(size = 10, hjust = 0.5))
    )
    
    # Save plot
    ggsave("C:/Users/esteb/Downloads/MELT_analysis/medieval_pos_analysis_complete.png", final_plot, 
           width = 16, height = 12, dpi = 300, bg = "white")
    
    # Display plot
    print(final_plot)
    
  }, error = function(e) {
    cat("❌ Error creating plots:", e$message, "\n")
  })
  
  # Summary table
  cat("\n📊 SUMMARY STATISTICS:\n")
  summary_table <- df %>%
    group_by(model) %>%
    summarise(
      n = n(),
      mean = round(mean(accuracy), 4),
      sd = round(sd(accuracy), 4),
      min = round(min(accuracy), 4),
      max = round(max(accuracy), 4),
      .groups = 'drop'
    )
  print(summary_table)
  
  # Key findings
  cat("\n🏆 KEY FINDINGS:\n")
  best_model <- summary_table$model[which.max(summary_table$mean)]
  cat("• Best model:", best_model, sprintf("(%.1f%% accuracy)\n", max(summary_table$mean) * 100))
  
  if (!is.null(stats_results$model_analysis)) {
    cat("• Model differences: Check ANOVA results above\n")
  }
  if (!is.null(stats_results$prompting_analysis)) {
    cat("• Prompting strategy: Check t-test results above\n")
  }
  if (!is.null(stats_results$dataset_analysis)) {
    cat("• Dataset difficulty: Check ANOVA results above\n")
  }
  
  cat("\n✅ Analysis complete!\n")
  cat("📁 Saved: medieval_pos_analysis_complete.png\n")
  
  return(list(data = df, results = stats_results, plot = if(exists("final_plot")) final_plot else NULL))
}

# ============================================================================
# RUN THE ANALYSIS
# ============================================================================

# Execute the complete analysis
analysis_results <- run_complete_analysis()

# Optional: Save workspace
save.image("C:/Users/esteb/Downloads/MELT_analysis/pos_tagging_analysis_workspace.RData")
cat("💾 Workspace saved: pos_tagging_analysis_workspace.RData\n")

cat("\n🎯 PRACTICAL RECOMMENDATIONS:\n")
cat("1. Check model performance in the analysis above\n")
cat("2. Review prompting strategy effectiveness\n") 
cat("3. Consider dataset-specific optimizations\n")
cat("4. Focus on statistically significant differences\n")