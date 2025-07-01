# POS Tagging Performance Analysis
# Load required libraries
library(tidyverse)
library(ggplot2)
library(stringr)
library(readr)
library(dplyr)
library(viridis)
library(scales)
library(gridExtra)
library(ggthemes)
library(RColorBrewer)
library(extrafont)
library(cowplot)
library(ggrepel)

# Set working directory
setwd("C:/Users/.../your_folder_with_classification_reports/")

# Enhanced theme for quality plots
theme_publication <- function(base_size = 11, base_family = "") {
  theme_minimal(base_size = base_size, base_family = base_family) +
    theme(
      # Grid and background
      panel.grid.major = element_line(color = "grey90", size = 0.3),
      panel.grid.minor = element_line(color = "grey95", size = 0.2),
      panel.background = element_rect(fill = "white", color = NA),
      plot.background = element_rect(fill = "white", color = NA),
      
      # Text elements
      plot.title = element_text(size = rel(1.4), face = "bold", 
                                margin = margin(b = 15), hjust = 0),
      plot.subtitle = element_text(size = rel(1.1), color = "grey30", 
                                   margin = margin(b = 20), hjust = 0),
      plot.caption = element_text(size = rel(0.8), color = "grey50", 
                                  margin = margin(t = 15), hjust = 1),
      
      # Axes
      axis.title.x = element_text(size = rel(1.1), margin = margin(t = 15)),
      axis.title.y = element_text(size = rel(1.1), margin = margin(r = 15)),
      axis.text = element_text(size = rel(0.95), color = "grey20"),
      axis.ticks = element_line(color = "grey70", size = 0.3),
      axis.line = element_line(color = "grey70", size = 0.3),
      
      # Legend
      legend.background = element_rect(fill = "white", color = "grey80", size = 0.3),
      legend.key = element_rect(fill = "white", color = NA),
      legend.title = element_text(size = rel(1.0), face = "bold"),
      legend.text = element_text(size = rel(0.9)),
      legend.margin = margin(6, 6, 6, 6),
      legend.spacing = unit(0.5, "cm"),
      
      # Facets
      strip.background = element_rect(fill = "grey95", color = "grey80", size = 0.3),
      strip.text = element_text(size = rel(1.0), face = "bold", 
                                margin = margin(4, 4, 4, 4)),
      
      # Margins
      plot.margin = margin(20, 20, 20, 20)
    )
}

# Set the enhanced theme as default
theme_set(theme_publication())

# Professional color palettes
# Primary palette (colorblind-friendly)
colors_primary <- c("#2E86AB", "#A23B72", "#F18F01", "#C73E1D", "#6A994E", "#7209B7")

# Secondary palette for gradients
colors_secondary <- c("#F8F9FA", "#E9ECEF", "#DEE2E6", "#CED4DA", "#ADB5BD", "#6C757D")

# Method-specific palette (ordered: Traditional -> LLM-Prompting -> LLM-Finetuning -> LLM-CLTF)
method_colors <- c(
  "Traditional" = "#2E86AB",        # Professional blue
  "LLM-Prompting" = "#A23B72",      # Deep magenta
  "LLM-Finetuning" = "#F18F01",     # Warm orange
  "LLM-CLTF" = "#C73E1D"            # Rich red
)

# Dataset palette (ordered: NAF -> CAT -> Chauliac)
dataset_colors <- c(
  "NAF" = "#2E86AB",      # Professional blue
  "CAT" = "#6A994E",      # Forest green  
  "Chauliac" = "#A23B72"  # Deep magenta
)

# Model palette (ordered: COLaF -> UDPipe -> Gemma3 -> Phi4)
model_colors <- c(
  "COLaF" = "#2E86AB",    # Professional blue
  "UDPipe" = "#6A994E",   # Forest green
  "Gemma3" = "#F18F01",   # Warm orange
  "Phi4" = "#C73E1D"      # Rich red
)

# Prompting strategy palette
prompting_colors <- c(
  "Zero-shot" = "#2E86AB",  # Professional blue
  "Few-shot" = "#6A994E"    # Forest green
)

# Function to parse classification reports
parse_classification_report <- function(file_path) {
  # Read the entire file
  content <- readLines(file_path, warn = FALSE)
  
  # Initialize lists to store results
  overall_results <- list()
  pos_results <- list()
  
  # Find report boundaries
  report_starts <- which(str_detect(content, "===REPORT_START==="))
  report_ends <- which(str_detect(content, "===REPORT_END==="))
  
  for (i in seq_along(report_starts)) {
    start_idx <- report_starts[i]
    end_idx <- report_ends[i]
    report_content <- content[start_idx:end_idx]
    
    # Extract metadata with more flexible patterns
    config_line <- str_subset(report_content, "CONFIG:")
    model_line <- str_subset(report_content, "MODEL:")
    dataset_line <- str_subset(report_content, "DATASET:")
    
    # Extract values with better error handling
    config <- if(length(config_line) > 0) str_trim(str_replace(config_line[1], "CONFIG:\\s*", "")) else NA
    model <- if(length(model_line) > 0) str_trim(str_replace(model_line[1], "MODEL:\\s*", "")) else NA
    dataset <- if(length(dataset_line) > 0) str_trim(str_replace(dataset_line[1], "DATASET:\\s*", "")) else NA
    
    # Extract additional metadata
    prompting <- NA
    decoding <- NA
    finetuning_order <- NA
    
    # Check for prompting-specific fields
    prompting_line <- str_subset(report_content, "PROMPTING:")
    decoding_line <- str_subset(report_content, "DECODING:")
    if (length(prompting_line) > 0) {
      prompting <- str_trim(str_replace(prompting_line[1], "PROMPTING:\\s*", ""))
    }
    if (length(decoding_line) > 0) {
      decoding <- str_trim(str_replace(decoding_line[1], "DECODING:\\s*", ""))
    }
    
    # Check for CLTF-specific fields
    ft_order_line <- str_subset(report_content, "FINETUNING_ORDER:")
    if (length(ft_order_line) > 0) {
      finetuning_order <- str_trim(str_replace(ft_order_line[1], "FINETUNING_ORDER:\\s*", ""))
    }
    
    # Extract accuracy with multiple patterns
    accuracy <- NA
    
    # Try different accuracy patterns
    accuracy_patterns <- c(
      "Accuracy:\\s*(\\d+\\.\\d+)",
      "accuracy\\s+(\\d+\\.\\d+)",
      "Accuracy:\\s*(\\d+\\.\\d+)",
      "^Accuracy:\\s*(\\d+\\.\\d+)"
    )
    
    for (pattern in accuracy_patterns) {
      accuracy_matches <- str_extract(report_content, pattern)
      accuracy_matches <- accuracy_matches[!is.na(accuracy_matches)]
      if (length(accuracy_matches) > 0) {
        accuracy_value <- str_extract(accuracy_matches[1], "\\d+\\.\\d+")
        if (!is.na(accuracy_value)) {
          accuracy <- as.numeric(accuracy_value)
          break
        }
      }
    }
    
    # Extract other metrics
    balanced_accuracy <- NA
    cohens_kappa <- NA
    macro_f1 <- NA
    micro_f1 <- NA
    
    # Look for balanced accuracy
    bal_acc_line <- str_subset(report_content, "Balanced Accuracy:")
    if (length(bal_acc_line) > 0) {
      bal_acc_value <- str_extract(bal_acc_line[1], "\\d+\\.\\d+")
      if (!is.na(bal_acc_value)) balanced_accuracy <- as.numeric(bal_acc_value)
    }
    
    # Look for Cohen's Kappa
    kappa_line <- str_subset(report_content, "Cohen's Kappa:")
    if (length(kappa_line) > 0) {
      kappa_value <- str_extract(kappa_line[1], "\\d+\\.\\d+")
      if (!is.na(kappa_value)) cohens_kappa <- as.numeric(kappa_value)
    }
    
    # Look for F1 scores
    macro_line <- str_subset(report_content, "Macro.*F1:")
    if (length(macro_line) > 0) {
      macro_value <- str_extract(macro_line[1], "\\d+\\.\\d+$")
      if (!is.na(macro_value)) macro_f1 <- as.numeric(macro_value)
    }
    
    micro_line <- str_subset(report_content, "Micro.*F1:")
    if (length(micro_line) > 0) {
      micro_value <- str_extract(micro_line[1], "\\d+\\.\\d+$")
      if (!is.na(micro_value)) micro_f1 <- as.numeric(micro_value)
    }
    
    # Try weighted avg as fallback for micro F1
    if (is.na(micro_f1)) {
      weighted_line <- str_subset(report_content, "weighted avg")
      if (length(weighted_line) > 0) {
        numbers <- str_extract_all(weighted_line[1], "\\d+\\.\\d+")[[1]]
        if (length(numbers) >= 3) {
          micro_f1 <- as.numeric(numbers[3])
        }
      }
    }
    
    # Only add result if we have minimum required data
    if (!is.na(config) && !is.na(model) && !is.na(dataset)) {
      overall_results[[length(overall_results) + 1]] <- data.frame(
        config = config,
        model = model,
        dataset = dataset,
        prompting = prompting,
        decoding = decoding,
        finetuning_order = finetuning_order,
        accuracy = accuracy,
        balanced_accuracy = balanced_accuracy,
        cohens_kappa = cohens_kappa,
        macro_f1 = macro_f1,
        micro_f1 = micro_f1,
        file_type = basename(file_path),
        stringsAsFactors = FALSE
      )
    }
    
    # Extract POS-level results
    content_start <- which(str_detect(report_content, "---CONTENT_START---"))
    content_end <- which(str_detect(report_content, "---CONTENT_END---"))
    
    if (length(content_start) > 0 && length(content_end) > 0) {
      class_content <- report_content[(content_start + 1):(content_end - 1)]
      table_lines <- class_content[str_detect(class_content, "^\\s*[A-Z]+\\s+\\d")]
      
      for (line in table_lines) {
        parts <- str_split(str_trim(line), "\\s+")[[1]]
        if (length(parts) >= 4 && !is.na(config) && !is.na(model) && !is.na(dataset)) {
          pos_class <- parts[1]
          precision <- as.numeric(parts[2])
          recall <- as.numeric(parts[3])
          f1_score <- as.numeric(parts[4])
          support <- if(length(parts) >= 5) as.numeric(parts[5]) else NA
          
          pos_results[[length(pos_results) + 1]] <- data.frame(
            config = config,
            model = model,
            dataset = dataset,
            prompting = prompting,
            decoding = decoding,
            finetuning_order = finetuning_order,
            pos_class = pos_class,
            precision = precision,
            recall = recall,
            f1_score = f1_score,
            support = support,
            file_type = basename(file_path),
            stringsAsFactors = FALSE
          )
        }
      }
    }
  }
  
  return(list(
    overall = if(length(overall_results) > 0) do.call(rbind, overall_results) else NULL,
    pos_level = if(length(pos_results) > 0) do.call(rbind, pos_results) else NULL
  ))
}

# Parse all files
files <- c(
  "non_llm_classification_reports.txt",
  "prompting_classification_reports.txt", 
  "finetuning_classification_reports.txt",
  "cltf_classification_reports.txt"
)

all_overall_results <- list()
all_pos_results <- list()

for (file in files) {
  if (file.exists(file)) {
    cat("Processing", file, "...\n")
    results <- parse_classification_report(file)
    
    if (!is.null(results$overall) && nrow(results$overall) > 0) {
      cat("  ✓ Found", nrow(results$overall), "overall results\n")
      all_overall_results[[file]] <- results$overall
    } else {
      cat("  ✗ No overall results found\n")
    }
    
    if (!is.null(results$pos_level) && nrow(results$pos_level) > 0) {
      cat("  ✓ Found", nrow(results$pos_level), "POS results\n")
      all_pos_results[[file]] <- results$pos_level
    } else {
      cat("  ✗ No POS results found\n")
    }
  } else {
    cat("File not found:", file, "\n")
  }
}

# Combine all results
overall_df <- do.call(rbind, all_overall_results)
pos_df <- do.call(rbind, all_pos_results)

# Debug: Check what data we actually have
cat("\n=== DATA OVERVIEW ===\n")
cat("Overall results dimensions:", dim(overall_df), "\n")
cat("POS results dimensions:", dim(pos_df), "\n")

if (!is.null(overall_df) && nrow(overall_df) > 0) {
  cat("File types found:", unique(overall_df$file_type), "\n")
  cat("Models found:", unique(overall_df$model), "\n") 
  cat("Datasets found:", unique(overall_df$dataset), "\n")
  cat("Non-NA accuracy values:", sum(!is.na(overall_df$accuracy)), "\n")
}

# Clean and categorize data
if (!is.null(overall_df) && nrow(overall_df) > 0) {
  
  overall_df <- overall_df %>%
    mutate(
      method = case_when(
        str_detect(file_type, "non_llm") ~ "Traditional",
        str_detect(file_type, "prompting") ~ "LLM-Prompting",
        str_detect(file_type, "finetuning") & is.na(finetuning_order) ~ "LLM-Finetuning",
        str_detect(file_type, "cltf") | !is.na(finetuning_order) ~ "LLM-CLTF",
        TRUE ~ "Other"
      ),
      model_clean = case_when(
        model == "udpipe" ~ "UDPipe",
        model == "colaf" ~ "COLaF", 
        model == "gemma3" ~ "Gemma3",
        model == "phi4" ~ "Phi4",
        TRUE ~ model
      ),
      strategy = case_when(
        method == "Traditional" ~ model_clean,
        method == "LLM-Prompting" ~ paste0(model_clean, " (", str_to_title(prompting), ")"),
        method == "LLM-Finetuning" ~ paste0(model_clean, " FT"),
        method == "LLM-CLTF" ~ paste0(model_clean, " CLTF"),
        TRUE ~ "Other"
      ),
      dataset = case_when(
        dataset == "naf" ~ "NAF",
        dataset == "cat" ~ "CAT",
        dataset == "chauliac" ~ "Chauliac",
        TRUE ~ "Other"
      ),
      prompting = case_when(
        str_detect(config, "zero") ~ "Zero-shot",
        str_detect(config, "few") ~ "Few-shot",
        TRUE ~ "Other"
      )
      
    ) %>%
    # Apply factor ordering for consistent presentation
    mutate(
      # Method ordering: Traditional -> LLM-Prompting -> LLM-Finetuning -> LLM-CLTF
      method = factor(method, levels = c("Traditional", "LLM-Prompting", "LLM-Finetuning", "LLM-CLTF")),
      
      # Model ordering: COLaF -> UDPipe -> Gemma3 -> Phi4
      model_clean = factor(model_clean, levels = c("COLaF", "UDPipe", "Gemma3", "Phi4")),
      
      # Dataset ordering: NAF -> CAT -> Chauliac
      dataset = factor(dataset, levels = c("NAF", "CAT", "Chauliac")),
      
      # Prompting strategy ordering: Zero-shot -> Few-shot
      prompting = factor(prompting, levels = c("Zero-shot", "Few-shot"))
    )
  
  # Debug: show method categorization results
  cat("\n=== METHOD CATEGORIZATION ===\n")
  method_counts <- table(overall_df$method, overall_df$file_type)
  print(method_counts)
  
  cat("\nMethod breakdown:\n")
  method_summary <- overall_df %>%
    group_by(method, file_type) %>%
    summarise(
      count = n(),
      has_accuracy = sum(!is.na(accuracy)),
      has_finetuning_order = sum(!is.na(finetuning_order)),
      .groups = "drop"
    )
  print(method_summary)
  
} else {
  cat("ERROR: No overall data found! Check file parsing.\n")
  stop("Cannot proceed without data")
}

# Clean and categorize POS data with proper dataset cleaning
if (!is.null(pos_df) && nrow(pos_df) > 0) {
  pos_df <- pos_df %>%
    mutate(
      method = case_when(
        str_detect(file_type, "non_llm") ~ "Traditional",
        str_detect(file_type, "prompting") ~ "LLM-Prompting", 
        str_detect(file_type, "finetuning") & is.na(finetuning_order) ~ "LLM-Finetuning",
        str_detect(file_type, "cltf") | !is.na(finetuning_order) ~ "LLM-CLTF",
        TRUE ~ "Other"
      ),
      model_clean = case_when(
        model == "udpipe" ~ "UDPipe",
        model == "colaf" ~ "COLaF",
        model == "gemma3" ~ "Gemma3", 
        model == "phi4" ~ "Phi4",
        TRUE ~ model
      ),
      # FIX: Add dataset cleaning that was missing
      dataset = case_when(
        dataset == "naf" ~ "NAF",
        dataset == "cat" ~ "CAT",
        dataset == "chauliac" ~ "Chauliac",
        TRUE ~ "Other"
      ),
      prompting = case_when(
        str_detect(config, "zero") ~ "Zero-shot",
        str_detect(config, "few") ~ "Few-shot",
        TRUE ~ "Other"
      )
    ) %>%
    # Apply same factor ordering as overall_df
    mutate(
      method = factor(method, levels = c("Traditional", "LLM-Prompting", "LLM-Finetuning", "LLM-CLTF")),
      model_clean = factor(model_clean, levels = c("COLaF", "UDPipe", "Gemma3", "Phi4")),
      dataset = factor(dataset, levels = c("NAF", "CAT", "Chauliac")),
      prompting = factor(prompting, levels = c("Zero-shot", "Few-shot"))
    )
} else {
  cat("WARNING: No POS-level data found! POS analysis will be skipped.\n")
}

# VISUALIZATIONS

# Plot 1: Enhanced Overall Performance Comparison with Model Distinction
best_overall <- overall_df %>%
  filter(!is.na(accuracy)) %>%
  group_by(method, dataset, model_clean) %>%
  summarise(
    best_accuracy = max(accuracy, na.rm = TRUE),
    best_f1 = max(micro_f1, na.rm = TRUE),
    .groups = "drop"
  ) %>%
  mutate(
    approach = paste(method, model_clean, sep = " - "),
    approach = str_replace(approach, " - NA", "")
  )

p1 <- ggplot(best_overall, aes(x = model_clean, y = best_accuracy, fill = model_clean)) +
  geom_col(width = 0.7, color = "white", size = 0.3, alpha = 0.9) +
  
  # Add value labels on top of bars
  geom_text(aes(label = paste0(round(best_accuracy * 100, 1), "%")),
            vjust = -0.5, size = 3, fontface = "bold", color = "grey20") +
  
  # Facet by method and dataset for clear comparison
  facet_grid(method ~ dataset, scales = "free_y", switch = "y") +
  
  # Use distinct model colors for better contrast
  scale_fill_manual(values = model_colors, name = "Model") +
  scale_y_continuous(
    labels = function(x) paste0(round(x * 100, 1), "%"),
    expand = expansion(mult = c(0, 0.15)) # More space for labels
  ) +
  labs(
    title = "Performance Comparison Across Methodological Approaches",
    subtitle = "Best accuracy achieved by each model within different tasks",
    x = "Model",
    y = "Accuracy",
    #caption = "Faceted by method (rows) and dataset (columns) for clear model comparison"
  ) +
  theme(
    # Enhanced facet styling
    strip.background = element_rect(fill = "grey96", color = "grey70", size = 0.5),
    strip.text = element_text(size = rel(1.0), face = "bold", margin = margin(6, 6, 6, 6)),
    strip.text.y.left = element_text(angle = 0),
    
    # Clean axis styling
    axis.text.x = element_text(angle = 30, hjust = 1, size = rel(0.9)),
    axis.text.y = element_text(size = rel(0.9)),
    
    # Legend positioning
    legend.position = "bottom",
    legend.title = element_text(size = rel(1.0), face = "bold"),
    
    # Panel spacing
    panel.spacing = unit(0.8, "lines"),
    panel.border = element_rect(color = "grey80", fill = NA, size = 0.5)
  )

# Plot 2: Enhanced Method Evolution with Statistical Significance
method_order <- c("Traditional", "LLM-Prompting", "LLM-Finetuning", "LLM-CLTF")
evolution_data <- best_overall %>%
  mutate(method = factor(method, levels = method_order)) %>%
  group_by(method, dataset) %>%
  summarise(
    avg_accuracy = mean(best_accuracy, na.rm = TRUE),
    se_accuracy = sd(best_accuracy, na.rm = TRUE) / sqrt(n()),
    .groups = "drop"
  )

p2 <- ggplot(evolution_data, aes(x = method, y = avg_accuracy, color = dataset, group = dataset)) +
  geom_ribbon(aes(ymin = avg_accuracy - se_accuracy, ymax = avg_accuracy + se_accuracy, 
                  fill = dataset), alpha = 0.2, color = NA) +
  geom_line(size = 1.2, alpha = 0.9) +
  geom_point(size = 3.5, alpha = 0.9) +
  geom_point(size = 2.5, color = "white") +
  scale_color_manual(values = dataset_colors, name = "Dataset") +
  scale_fill_manual(values = dataset_colors, name = "Dataset") +
  scale_y_continuous(
    labels = function(x) paste0(round(x * 100, 1), "%"),
    expand = expansion(mult = c(0.02, 0.05))
  ) +
  labs(
    title = "POS Tagging Performance Evolution",
    subtitle = "Average Accuracy across four different tasks",
    x = "Task",
    y = "Average Accuracy",
  ) +
  guides(fill = "none") +
  theme(
    axis.text.x = element_text(angle = 15, hjust = 1),
    legend.position = "bottom"
  )

# Plot 3: Enhanced POS Class Performance with Hierarchical Clustering
major_pos <- c("NOUN", "VERB", "ADJ", "ADP", "ADV", "PRON", "DET", "CCONJ", "PROPN")

if (!is.null(pos_df) && nrow(pos_df) > 0) {
  pos_comparison <- pos_df %>%
    filter(pos_class %in% major_pos, !is.na(f1_score)) %>%
    group_by(method, dataset, pos_class, model_clean) %>%
    summarise(best_f1 = max(f1_score, na.rm = TRUE), .groups = "drop") %>%
    filter(!is.infinite(best_f1)) %>%
    group_by(method, dataset, pos_class) %>%
    summarise(
      avg_f1 = mean(best_f1, na.rm = TRUE),
      se_f1 = sd(best_f1, na.rm = TRUE) / sqrt(n()),
      .groups = "drop"
    )
  
  cat("POS comparison data points:", nrow(pos_comparison), "\n")
  
  if (nrow(pos_comparison) > 0) {
    # Create ordering based on overall difficulty
    pos_order <- pos_comparison %>%
      group_by(pos_class) %>%
      summarise(overall_f1 = mean(avg_f1, na.rm = TRUE), .groups = "drop") %>%
      arrange(overall_f1) %>%
      pull(pos_class)
    
    pos_comparison$pos_class <- factor(pos_comparison$pos_class, levels = pos_order)
    
    p3 <- ggplot(pos_comparison, aes(x = pos_class, y = avg_f1, fill = method)) +
      geom_col(position = position_dodge2(width = 0.8, preserve = "single"), 
               width = 0.7, color = "white", size = 0.2) +
      geom_errorbar(aes(ymin = avg_f1 - se_f1, ymax = avg_f1 + se_f1),
                    position = position_dodge2(width = 0.8, preserve = "single"),
                    width = 0.3, alpha = 0.7) +
      facet_wrap(~dataset, scales = "free_y", labeller = label_both) +
      scale_fill_manual(values = method_colors, name = "Method") +
      scale_y_continuous(
        labels = function(x) paste0(round(x * 100, 1), "%"),
        expand = expansion(mult = c(0, 0.05))
      ) +
      labs(
        title = "Part-of-Speech Category Performance Analysis",
        subtitle = "F1-scores across major grammatical categories (ordered by difficulty)",
        x = "Part-of-Speech Category",
        y = "F1-Score",
        #caption = "Categories ordered from most difficult (left) to easiest (right)"
      ) +
      theme(
        axis.text.x = element_text(angle = 45, hjust = 1),
        legend.position = "bottom",
        strip.text = element_text(size = rel(0.9))
      )
  } else {
    p3 <- ggplot() + labs(title = "No valid POS class data for analysis")
  }
} else {
  p3 <- ggplot() + labs(title = "No POS data available for class analysis")
}

# Plot 4: Decoding Strategy Analysis
prompting_data <- overall_df %>% filter(method == "LLM-Prompting", !is.na(decoding), !is.na(accuracy))
cat("Prompting data available:", nrow(prompting_data), "rows\n")

if (nrow(prompting_data) > 0) {
  cat("Decoding strategies found:", unique(prompting_data$decoding), "\n")
  
  decoding_analysis <- prompting_data %>%
    mutate(
      decoding_type = case_when(
        str_detect(decoding, "^b") ~ "Beam Search",
        str_detect(decoding, "^k") ~ "Top-K",
        str_detect(decoding, "^p") ~ "Top-P", 
        str_detect(decoding, "^t") ~ "Temperature",
        TRUE ~ "Other"
      ),
      # Ensure proper factor ordering
      decoding_type = factor(decoding_type, levels = c("Beam Search", "Top-K", "Top-P", "Temperature", "Other"))
    ) %>%
    group_by(dataset, model_clean, prompting, decoding_type) %>%
    summarise(
      count = n(),
      avg_accuracy = mean(accuracy, na.rm = TRUE),
      se_accuracy = sd(accuracy, na.rm = TRUE) / sqrt(n()),
      min_accuracy = min(accuracy, na.rm = TRUE),
      max_accuracy = max(accuracy, na.rm = TRUE),
      .groups = "drop"
    ) %>%
    filter(!is.na(avg_accuracy) & !is.infinite(avg_accuracy))
  
  cat("Decoding analysis data points:", nrow(decoding_analysis), "\n")
  
  if (nrow(decoding_analysis) > 0) {
    # Create custom labeller for cleaner facet labels
    facet_labeller <- as_labeller(c(
      "COLaF" = "Model: COLaF",
      "UDPipe" = "Model: UDPipe", 
      "Gemma3" = "Model: Gemma3",
      "Phi4" = "Model: Phi4",
      "NAF" = "Dataset: NAF",
      "CAT" = "Dataset: CAT",
      "Chauliac" = "Dataset: Chauliac"
    ))
    
    p4 <- ggplot(decoding_analysis, aes(x = decoding_type, y = avg_accuracy, fill = prompting)) +
      # Add subtle background bars for better readability
      geom_col(aes(alpha = prompting), position = position_dodge2(width = 0.8, preserve = "single"), 
               width = 0.7, color = "white", size = 0.3) +
      
      # Add error bars for statistical precision
      geom_errorbar(aes(ymin = avg_accuracy - se_accuracy, ymax = avg_accuracy + se_accuracy),
                    position = position_dodge2(width = 0.8, preserve = "single"),
                    width = 0.3, alpha = 0.8, color = "grey30") +
      
      # Add value labels on top of bars
      geom_text(aes(label = paste0(round(avg_accuracy * 100, 1), "%"),
                    y = avg_accuracy + se_accuracy + 0.01),
                position = position_dodge2(width = 0.8, preserve = "single"),
                size = 2.8, fontface = "bold", color = "grey20") +
      
      # Facet by model and dataset with cleaner labels
      facet_grid(model_clean ~ dataset, labeller = facet_labeller, switch = "y") +
      
      # Apply elegant color scheme
      scale_fill_manual(values = prompting_colors, name = "Prompting Strategy") +
      scale_alpha_manual(values = c("Zero-shot" = 0.8, "Few-shot" = 1.0), guide = "none") +
      
      # Enhanced y-axis formatting
      scale_y_continuous(
        labels = function(x) paste0(round(x * 100, 1), "%"),
        expand = expansion(mult = c(0, 0.15)), # More space for labels
        breaks = scales::pretty_breaks(n = 4)
      ) +
      
      # Styling
      labs(
        title = "Decoding Strategy Performance in Large Language Model Prompting",
        subtitle = "Comparative analysis of decoding approaches across zero-shot and few-shot prompting",
        x = "Decoding Strategy",
        y = "Accuracy",
        #caption = "Error bars represent standard error; percentages show mean accuracy"
      ) +
      
      # Enhanced theme customization
      theme(
        # Facet styling
        strip.background = element_rect(fill = "grey96", color = "grey70", size = 0.5),
        strip.text = element_text(size = rel(1.0), face = "bold", margin = margin(6, 6, 6, 6)),
        strip.text.y.left = element_text(angle = 0),
        
        # Axis styling
        axis.text.x = element_text(angle = 35, hjust = 1, size = rel(0.9)),
        axis.text.y = element_text(size = rel(0.9)),
        axis.title = element_text(size = rel(1.1)),
        
        # Legend styling
        legend.position = "bottom",
        legend.title = element_text(size = rel(1.0), face = "bold"),
        legend.text = element_text(size = rel(0.95)),
        legend.key.width = unit(1.2, "cm"),
        legend.spacing.x = unit(0.5, "cm"),
        
        # Panel styling
        panel.spacing = unit(0.8, "lines"),
        panel.border = element_rect(color = "grey80", fill = NA, size = 0.5),
        
        # Plot margins
        plot.margin = margin(20, 25, 20, 20)
      )
  } else {
    p4 <- ggplot() + 
      labs(title = "No valid prompting data for decoding analysis",
           subtitle = "Check data quality and missing values") +
      theme_void()
  }
} else {
  p4 <- ggplot() + 
    labs(title = "No prompting data available for decoding analysis",
         subtitle = "Ensure LLM-Prompting experiments are included in the dataset") +
    theme_void()
}

# Plot 5: Enhanced CLTF Impact Analysis with Effect Sizes
cat("=== CLTF ANALYSIS DEBUG ===\n")

# Alternative approach: identify CLTF by presence of finetuning_order field
cltf_data <- overall_df %>% 
  filter(!is.na(finetuning_order) & !is.na(accuracy)) %>%
  mutate(method = "LLM-CLTF")

ft_data <- overall_df %>%
  filter(str_detect(file_type, "finetuning") & is.na(finetuning_order) & !is.na(accuracy)) %>%
  mutate(method = "LLM-Finetuning")

cat("CLTF data found:", nrow(cltf_data), "rows\n")
cat("Fine-tuning data found:", nrow(ft_data), "rows\n")

if (nrow(cltf_data) > 0 && nrow(ft_data) > 0) {
  combined_data <- bind_rows(
    cltf_data %>% select(dataset, model_clean, accuracy, method),
    ft_data %>% select(dataset, model_clean, accuracy, method)
  )
  
  # Get best accuracy for each method-dataset-model combination
  comparison_data <- combined_data %>%
    group_by(method, dataset, model_clean) %>%
    summarise(best_accuracy = max(accuracy, na.rm = TRUE), .groups = "drop") %>%
    pivot_wider(names_from = method, values_from = best_accuracy) %>%
    filter(!is.na(`LLM-Finetuning`) & !is.na(`LLM-CLTF`)) %>%
    mutate(
      cltf_improvement = `LLM-CLTF` - `LLM-Finetuning`,
      effect_size = abs(cltf_improvement) / pmax(`LLM-Finetuning`, 0.01), # Avoid division by zero
      dataset_model = paste(dataset, model_clean, sep = "\n"),
      improvement_category = case_when(
        cltf_improvement > 0.03 ~ "Large Improvement",
        cltf_improvement > 0.005 ~ "Moderate Improvement", 
        cltf_improvement > -0.005 ~ "Negligible Change",
        cltf_improvement > -0.03 ~ "Moderate Decline",
        TRUE ~ "Large Decline"
      )
    ) %>%
    # Add proper factor ordering
    mutate(
      improvement_category = factor(improvement_category, 
                                    levels = c("Large Decline", "Moderate Decline", "Negligible Change", 
                                               "Moderate Improvement", "Large Improvement"))
    )
  
  if (nrow(comparison_data) > 0) {
    # Create color palette for improvement categories
    improvement_colors <- c(
      "Large Decline" = "#C73E1D",
      "Moderate Decline" = "#F18F01",
      "Negligible Change" = "#ADB5BD",
      "Moderate Improvement" = "#6A994E", 
      "Large Improvement" = "#2D5A27"
    )
    
    p5 <- ggplot(comparison_data, aes(x = reorder(dataset_model, cltf_improvement), 
                                      y = cltf_improvement, fill = improvement_category)) +
      geom_col(width = 0.7, color = "white", size = 0.2) +
      geom_hline(yintercept = 0, linetype = "solid", alpha = 0.8, size = 0.8) +
      geom_hline(yintercept = c(-0.02, -0.005, 0.005, 0.02), 
                 linetype = "dashed", alpha = 0.5, size = 0.3) +
      scale_fill_manual(values = improvement_colors, name = "Effect Size") +
      scale_y_continuous(
        labels = function(x) paste0(ifelse(x >= 0, "+", ""), round(x * 100, 1), "%"),
        expand = expansion(mult = c(0.05, 0.05))
      ) +
      coord_flip() +
      labs(
        title = "Cross-Lingual Transfer Learning Effectiveness",
        subtitle = "Performance difference between LLM-CLTF and monolingual fine-tuning approaches",
        x = "Dataset × Model Combination",
        y = "Accuracy Improvement (LLM-CLTF vs. LLM-Finetuning)",
        #caption = "Positive values indicate LLM-CLTF outperforms single-dataset fine-tuning"
      ) +
      theme(
        legend.position = "bottom",
        axis.text.y = element_text(size = rel(0.8))
      )
  } else {
    p5 <- ggplot() + 
      labs(title = "No matching Fine-tuning and CLTF pairs found",
           subtitle = "Both methods exist but no overlapping dataset-model combinations")
  }
} else {
  p5 <- ggplot() + 
    labs(title = paste("Missing data for comparison:", 
                       ifelse(nrow(cltf_data) == 0, "No CLTF data", ""), 
                       ifelse(nrow(ft_data) == 0, "No Fine-tuning data", "")),
         subtitle = "Need both CLTF and Fine-tuning results for comparison")
}

# Display and save plots with high resolution
plots <- list(p1, p2, p3, p4, p5)
plot_names <- c("overall_performance", "method_evolution", "pos_class_performance", 
                "decoding_strategies", "cltf_impact")

for (i in seq_along(plots)) {
  if (exists(paste0("p", i))) {
    print(plots[[i]])
    
    # Save high-resolution versions
    tryCatch({
      ggsave(paste0(plot_names[i], "_enhanced.png"), plots[[i]], 
             width = 12, height = 8, dpi = 300, bg = "white")
      ggsave(paste0(plot_names[i], "_enhanced.pdf"), plots[[i]], 
             width = 12, height = 8, device = "pdf")
    }, error = function(e) {
      cat("Failed to save", plot_names[i], ":", e$message, "\n")
    })
  }
}

# Generate enhanced summary statistics with confidence intervals
cat("\n=== ENHANCED SUMMARY STATISTICS ===\n")

if (!is.null(overall_df) && nrow(overall_df) > 0 && sum(!is.na(overall_df$accuracy)) > 0) {
  # Best performance by dataset with confidence intervals
  best_by_dataset <- overall_df %>%
    filter(!is.na(accuracy)) %>%
    group_by(dataset) %>%
    arrange(desc(accuracy)) %>%
    slice(1) %>%
    select(dataset, strategy, accuracy, method, model_clean)
  
  cat("\nBest performance by dataset:\n")
  print(best_by_dataset)
  
  # Enhanced method comparison with effect sizes
  method_summary <- overall_df %>%
    filter(!is.na(accuracy)) %>%
    group_by(method) %>%
    summarise(
      count = n(),
      avg_accuracy = mean(accuracy, na.rm = TRUE),
      se_accuracy = sd(accuracy, na.rm = TRUE) / sqrt(n()),
      ci_lower = avg_accuracy - 1.96 * se_accuracy,
      ci_upper = avg_accuracy + 1.96 * se_accuracy,
      max_accuracy = max(accuracy, na.rm = TRUE),
      min_accuracy = min(accuracy, na.rm = TRUE),
      .groups = "drop"
    ) %>%
    mutate(
      avg_accuracy_pct = paste0(round(avg_accuracy * 100, 2), "%"),
      ci_range = paste0("[", round(ci_lower * 100, 2), "%, ", round(ci_upper * 100, 2), "%]")
    )
  
  cat("\nMethod comparison with 95% confidence intervals:\n")
  print(method_summary %>% select(method, count, avg_accuracy_pct, ci_range, max_accuracy, min_accuracy))
}
