# ---------------------------------------------------------
# fig20_weights3.R
# How the fusion weight divides across three modalities
# ---------------------------------------------------------

library(ggplot2)
library(dplyr)
library(tidyr)
library(grid)
library(ragg)
library(systemfonts)

## ---- 1. house settings ----------------------------------

fig_dir <- paste0(paste0(Sys.getenv("USERPROFILE"), "/Documents/"),
                  "Data Science/Dissertation/R graphs")
dir.create(fig_dir, recursive = TRUE,
           showWarnings = FALSE)

data_dir <- paste0(paste0(Sys.getenv("USERPROFILE"), "/Documents/"),
                   "Data Science/Dissertation/figwork/data")

FONT <- "Calibri"
if (nrow(subset(system_fonts(),
                family == FONT)) == 0) {
  message("Calibri not found - falling back to sans")
  FONT <- "sans"
}

theme_set(theme_minimal(base_size = 11,
                        base_family = FONT))

save_fig <- function(plot, name, width, height) {
  p_png <- file.path(fig_dir, paste0(name, ".png"))
  p_pdf <- file.path(fig_dir, paste0(name, ".pdf"))
  ggsave(p_png, plot, device = agg_png,
         width = width, height = height,
         units = "in", dpi = 400, bg = "white")
  ggsave(p_pdf, plot, device = cairo_pdf,
         width = width, height = height,
         units = "in", bg = "white")
  message("saved: ", p_png)
  message("saved: ", p_pdf)
}

OUT_LAB <- c(composite_30d = "Composite (30-Day)",
             death_30d = "30-Day Death",
             cv_first = "CV Readmission")

ARM_LAB <- c(w_ehr = "EHR",
             w_ecg = "ECG",
             w_ctpa = "CTPA")

## Okabe-Ito, matching the modality colours used throughout
PAL <- c("EHR"  = "#0072B2",
         "ECG"  = "#E69F00",
         "CTPA" = "#009E73")

## darker variants, for text on a white plate
PAL_TXT <- c("EHR"  = "#00446B",
             "ECG"  = "#8A5F00",
             "CTPA" = "#00604A")

## ---- 2. weights ----------------------------------------

w <- read.csv(file.path(data_dir,
                        "wmean3_weights.csv"),
              stringsAsFactors = FALSE)
cat("\n--- fold weights ---\n")
print(w, digits = 2)

keep <- intersect(names(OUT_LAB), unique(w$outcome))
ORD <- rev(keep)

mw <- w %>%
  filter(outcome %in% keep) %>%
  group_by(outcome) %>%
  summarise(across(c(w_ehr, w_ecg, w_ctpa), mean),
            .groups = "drop")

stopifnot(all(abs(
  mw$w_ehr + mw$w_ecg + mw$w_ctpa - 1) < 1e-8))

rng <- w %>%
  filter(outcome %in% keep) %>%
  group_by(outcome) %>%
  summarise(lo = min(w_ctpa), hi = max(w_ctpa),
            .groups = "drop") %>%
  mutate(lab = ifelse(
    lo == hi,
    sprintf("CTPA %.2f in every fold", lo),
    sprintf("CTPA %.2f\u2013%.2f across folds",
            lo, hi)),
    yy = match(as.character(outcome), ORD))

cat("\n--- CTPA weight range ---\n")
print(as.data.frame(rng))

d <- mw %>%
  pivot_longer(-outcome, names_to = "arm",
               values_to = "wt") %>%
  mutate(arm = factor(ARM_LAB[arm],
                      levels = unname(ARM_LAB))) %>%
  arrange(match(outcome, ORD), arm) %>%
  group_by(outcome) %>%
  mutate(hi = cumsum(wt), lo = hi - wt,
         mid = (lo + hi) / 2) %>%
  ungroup() %>%
  mutate(yy = match(as.character(outcome), ORD))

HH <- 0.30   # half-height of each bar

## ---- 3. pattern overlays for greyscale ------------------
## EHR solid; ECG vertical rules; CTPA stipple.

vlines <- do.call(rbind, lapply(
  which(d$arm == "ECG"), function(i) {
    r <- d[i, ]
    xs <- seq(r$lo + 0.008, r$hi - 0.008, by = 0.011)
    if (length(xs) < 1) return(NULL)
    data.frame(x = xs, yy = r$yy)
  }))

dots <- do.call(rbind, lapply(
  which(d$arm == "CTPA"), function(i) {
    r <- d[i, ]
    xs <- seq(r$lo + 0.010, r$hi - 0.010, by = 0.016)
    if (length(xs) < 1) return(NULL)
    expand.grid(x = xs,
                dy = c(-0.16, 0, 0.16)) %>%
      mutate(yy = r$yy + dy)
  }))

cat("\npattern overlays: ",
    ifelse(is.null(vlines), 0, nrow(vlines)),
    " rules, ",
    ifelse(is.null(dots), 0, nrow(dots)),
    " stipple points\n", sep = "")

lab_d <- filter(d, wt >= 0.08)
top <- d %>% filter(yy == max(yy))

## ---- 4. plot and save -----------------------------------

f <- ggplot() +
  geom_vline(xintercept = seq(0.2, 0.8, by = 0.2),
             colour = "grey90", linewidth = 0.3) +
  geom_rect(data = d,
            aes(xmin = lo, xmax = hi, fill = arm,
                ymin = yy - HH, ymax = yy + HH),
            colour = "white", linewidth = 0.7)

if (!is.null(vlines)) {
  f <- f + geom_segment(
    data = vlines,
    aes(x = x, xend = x,
        y = yy - HH + 0.02, yend = yy + HH - 0.02),
    colour = "white", linewidth = 0.35,
    inherit.aes = FALSE)
}
if (!is.null(dots)) {
  f <- f + geom_point(
    data = dots, aes(x = x, y = yy),
    colour = "white", size = 0.55,
    inherit.aes = FALSE)
}

f <- f +
  geom_label(data = lab_d,
             aes(x = mid, y = yy,
                 label = sprintf("%.2f", wt),
                 colour = arm),
             fill = "white", label.size = 0,
             label.r = unit(0.10, "lines"),
             label.padding = unit(0.16, "lines"),
             fontface = "bold", size = 3.4,
             family = FONT, show.legend = FALSE) +
  geom_text(data = top,
            aes(x = mid, y = yy + HH + 0.24,
                label = arm, colour = arm),
            fontface = "bold", size = 3.2,
            family = FONT, show.legend = FALSE) +
  geom_text(data = rng,
            aes(x = 1.02, y = yy, label = lab),
            hjust = 0, size = 3.1, colour = "grey30",
            family = FONT) +
  scale_fill_manual(values = PAL, guide = "none") +
  scale_colour_manual(values = PAL_TXT,
                      guide = "none") +
  scale_x_continuous(
    limits = c(0, 1.44),
    breaks = seq(0, 1, by = 0.2),
    labels = function(x) sprintf("%.1f", x),
    expand = c(0, 0)) +
  scale_y_continuous(
    breaks = seq_along(ORD),
    labels = OUT_LAB[ORD],
    limits = c(0.5, length(ORD) + 0.95),
    expand = c(0, 0)) +
  labs(x = paste("Share of the Fusion Weight",
                 "(Modalities Sum to 1.0)"),
       y = NULL,
       title = paste("How the Fusion Weight Divides",
                     "Across Three Modalities")) +
  theme(panel.grid = element_blank(),
        axis.text.y = element_text(size = 10),
        axis.text.x = element_text(size = 9.5),
        axis.title.x = element_text(
          size = 9.5, margin = margin(t = 8)),
        plot.title = element_text(
          size = 12.5, face = "bold", hjust = 0.5,
          margin = margin(b = 10)),
        plot.title.position = "plot",
        plot.margin = margin(6, 6, 4, 6))

print(f)
save_fig(f, "fig20_weights3", 8.4, 3.4)