//! Statistics Module for Advanced Mathematical Analysis
//!
//! This module implements comprehensive statistical operations including
//! descriptive statistics, probability distributions, hypothesis testing,
//! and regression analysis with robust numerical methods.

use crate::error::{NEATError, Result};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::f64::consts::{E, PI};

/// Types of statistical operations
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum StatisticOperation {
    /// Descriptive statistics (mean, median, mode, etc.)
    Descriptive,
    /// Probability calculations
    Probability,
    /// Distribution fitting and analysis
    Distribution,
    /// Hypothesis testing
    HypothesisTesting,
    /// Correlation and regression
    Regression,
    /// Time series analysis
    TimeSeries,
    /// Confidence intervals
    ConfidenceInterval,
    /// ANOVA analysis
    ANOVA,
}

/// Statistical distribution types
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum DistributionType {
    /// Normal (Gaussian) distribution
    Normal,
    /// Binomial distribution
    Binomial,
    /// Poisson distribution
    Poisson,
    /// Exponential distribution
    Exponential,
    /// Uniform distribution
    Uniform,
    /// Chi-squared distribution
    ChiSquared,
    /// Student's t-distribution
    TDistribution,
    /// F-distribution
    FDistribution,
}

/// Descriptive statistics summary
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DescriptiveStats {
    /// Sample size
    pub n: usize,
    /// Arithmetic mean
    pub mean: f64,
    /// Median (50th percentile)
    pub median: f64,
    /// Mode (most frequent value)
    pub mode: Option<f64>,
    /// Sample standard deviation
    pub std_dev: f64,
    /// Sample variance
    pub variance: f64,
    /// Minimum value
    pub min: f64,
    /// Maximum value
    pub max: f64,
    /// Range (max - min)
    pub range: f64,
    /// Interquartile range (Q3 - Q1)
    pub iqr: f64,
    /// First quartile (25th percentile)
    pub q1: f64,
    /// Third quartile (75th percentile)
    pub q3: f64,
    /// Skewness measure
    pub skewness: f64,
    /// Kurtosis measure
    pub kurtosis: f64,
    /// Standard error of the mean
    pub standard_error: f64,
}

/// Distribution parameters and properties
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DistributionInfo {
    /// Type of distribution
    pub distribution_type: DistributionType,
    /// Distribution parameters
    pub parameters: HashMap<String, f64>,
    /// Mean of the distribution
    pub mean: f64,
    /// Variance of the distribution
    pub variance: f64,
    /// Standard deviation
    pub std_dev: f64,
    /// Support (domain) of the distribution
    pub support: (f64, f64),
    /// Whether distribution is discrete or continuous
    pub is_discrete: bool,
}

/// Hypothesis test result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HypothesisTestResult {
    /// Test statistic value
    pub test_statistic: f64,
    /// P-value
    pub p_value: f64,
    /// Critical value(s)
    pub critical_values: Vec<f64>,
    /// Degrees of freedom (if applicable)
    pub degrees_of_freedom: Option<usize>,
    /// Significance level (alpha)
    pub alpha: f64,
    /// Whether to reject null hypothesis
    pub reject_null: bool,
    /// Test type description
    pub test_type: String,
    /// Effect size (if applicable)
    pub effect_size: Option<f64>,
}

/// Regression analysis result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RegressionResult {
    /// Regression coefficients
    pub coefficients: Vec<f64>,
    /// R-squared value
    pub r_squared: f64,
    /// Adjusted R-squared
    pub adjusted_r_squared: f64,
    /// Standard errors of coefficients
    pub standard_errors: Vec<f64>,
    /// T-statistics for coefficients
    pub t_statistics: Vec<f64>,
    /// P-values for coefficients
    pub p_values: Vec<f64>,
    /// Residual sum of squares
    pub residual_ss: f64,
    /// Total sum of squares
    pub total_ss: f64,
    /// F-statistic for overall model
    pub f_statistic: f64,
    /// Degrees of freedom
    pub df_model: usize,
    pub df_residual: usize,
}

/// Confidence interval result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConfidenceInterval {
    /// Lower bound
    pub lower_bound: f64,
    /// Upper bound
    pub upper_bound: f64,
    /// Confidence level (e.g., 0.95 for 95%)
    pub confidence_level: f64,
    /// Margin of error
    pub margin_of_error: f64,
    /// Statistic being estimated
    pub statistic: String,
}

/// Main statistics engine
pub struct StatisticsEngine {
    /// Numerical precision for computations
    precision: f64,
    /// Default significance level
    default_alpha: f64,
    /// Random number generator seed
    rng_seed: Option<u64>,
}

impl Default for StatisticsEngine {
    fn default() -> Self {
        Self {
            precision: 1e-12,
            default_alpha: 0.05,
            rng_seed: None,
        }
    }
}

impl StatisticsEngine {
    /// Create a new statistics engine
    pub fn new(precision: f64, default_alpha: f64) -> Self {
        Self {
            precision,
            default_alpha,
            rng_seed: None,
        }
    }

    /// Compute descriptive statistics for a dataset
    pub fn descriptive_statistics(&self, data: &[f64]) -> Result<DescriptiveStats> {
        if data.is_empty() {
            return Err(NEATError::InvalidConfiguration {
                parameter: "dataset".to_string(),
                value: "Empty dataset".to_string(),
            });
        }

        let n = data.len();
        let mut sorted_data = data.to_vec();
        sorted_data.sort_by(|a, b| a.partial_cmp(b).unwrap());

        // Basic statistics
        let mean = self.mean(data);
        let median = self.median(&sorted_data);
        let mode = self.mode(data);
        let variance = self.variance(data, mean);
        let std_dev = variance.sqrt();

        // Range statistics
        let min = sorted_data[0];
        let max = sorted_data[n - 1];
        let range = max - min;

        // Quartiles
        let q1 = self.percentile(&sorted_data, 25.0);
        let q3 = self.percentile(&sorted_data, 75.0);
        let iqr = q3 - q1;

        // Moments
        let skewness = self.skewness(data, mean, std_dev);
        let kurtosis = self.kurtosis(data, mean, std_dev);

        // Standard error
        let standard_error = std_dev / (n as f64).sqrt();

        Ok(DescriptiveStats {
            n,
            mean,
            median,
            mode,
            std_dev,
            variance,
            min,
            max,
            range,
            iqr,
            q1,
            q3,
            skewness,
            kurtosis,
            standard_error,
        })
    }

    /// Perform one-sample t-test
    pub fn one_sample_t_test(&self, data: &[f64], hypothesized_mean: f64, alpha: Option<f64>) -> Result<HypothesisTestResult> {
        let alpha = alpha.unwrap_or(self.default_alpha);
        let n = data.len();
        
        if n < 2 {
            return Err(NEATError::InvalidConfiguration {
                parameter: "sample_size".to_string(),
                value: "Need at least 2 observations for t-test".to_string(),
            });
        }

        let sample_mean = self.mean(data);
        let sample_std = self.variance(data, sample_mean).sqrt();
        let standard_error = sample_std / (n as f64).sqrt();
        
        // Calculate t-statistic
        let t_statistic = (sample_mean - hypothesized_mean) / standard_error;
        
        // Degrees of freedom
        let df = n - 1;
        
        // Calculate p-value (two-tailed)
        let p_value = 2.0 * (1.0 - self.t_cdf(t_statistic.abs(), df));
        
        // Critical value
        let t_critical = self.t_inverse(1.0 - alpha / 2.0, df);
        
        let reject_null = p_value < alpha;
        
        // Calculate effect size (Cohen's d)
        let effect_size = (sample_mean - hypothesized_mean) / sample_std;

        Ok(HypothesisTestResult {
            test_statistic: t_statistic,
            p_value,
            critical_values: vec![-t_critical, t_critical],
            degrees_of_freedom: Some(df),
            alpha,
            reject_null,
            test_type: "One-sample t-test".to_string(),
            effect_size: Some(effect_size),
        })
    }

    /// Perform two-sample t-test
    pub fn two_sample_t_test(&self, data1: &[f64], data2: &[f64], alpha: Option<f64>) -> Result<HypothesisTestResult> {
        let alpha = alpha.unwrap_or(self.default_alpha);
        let n1 = data1.len();
        let n2 = data2.len();
        
        if n1 < 2 || n2 < 2 {
            return Err(NEATError::InvalidConfiguration {
                parameter: "sample_sizes".to_string(),
                value: "Need at least 2 observations in each group".to_string(),
            });
        }

        let mean1 = self.mean(data1);
        let mean2 = self.mean(data2);
        let var1 = self.variance(data1, mean1);
        let var2 = self.variance(data2, mean2);
        
        // Welch's t-test (unequal variances)
        let pooled_se = (var1 / n1 as f64 + var2 / n2 as f64).sqrt();
        let t_statistic = (mean1 - mean2) / pooled_se;
        
        // Welch-Satterthwaite equation for degrees of freedom
        let numerator = (var1 / n1 as f64 + var2 / n2 as f64).powi(2);
        let denominator = (var1 / n1 as f64).powi(2) / (n1 - 1) as f64 + 
                         (var2 / n2 as f64).powi(2) / (n2 - 1) as f64;
        let df = (numerator / denominator).floor() as usize;
        
        let p_value = 2.0 * (1.0 - self.t_cdf(t_statistic.abs(), df));
        let t_critical = self.t_inverse(1.0 - alpha / 2.0, df);
        
        let reject_null = p_value < alpha;
        
        // Effect size (Cohen's d)
        let pooled_std = ((var1 * (n1 - 1) as f64 + var2 * (n2 - 1) as f64) / 
                         (n1 + n2 - 2) as f64).sqrt();
        let effect_size = (mean1 - mean2) / pooled_std;

        Ok(HypothesisTestResult {
            test_statistic: t_statistic,
            p_value,
            critical_values: vec![-t_critical, t_critical],
            degrees_of_freedom: Some(df),
            alpha,
            reject_null,
            test_type: "Two-sample t-test (Welch)".to_string(),
            effect_size: Some(effect_size),
        })
    }

    /// Perform simple linear regression
    pub fn linear_regression(&self, x: &[f64], y: &[f64]) -> Result<RegressionResult> {
        if x.len() != y.len() {
            return Err(NEATError::InvalidConfiguration {
                parameter: "data_length".to_string(),
                value: "X and Y must have same length".to_string(),
            });
        }
        
        let n = x.len();
        if n < 3 {
            return Err(NEATError::InvalidConfiguration {
                parameter: "sample_size".to_string(),
                value: "Need at least 3 observations for regression".to_string(),
            });
        }

        let x_mean = self.mean(x);
        let y_mean = self.mean(y);
        
        // Calculate regression coefficients
        let mut numerator = 0.0;
        let mut denominator = 0.0;
        
        for i in 0..n {
            let x_diff = x[i] - x_mean;
            let y_diff = y[i] - y_mean;
            numerator += x_diff * y_diff;
            denominator += x_diff * x_diff;
        }
        
        if denominator.abs() < self.precision {
            return Err(NEATError::InvalidConfiguration {
                parameter: "regression".to_string(),
                value: "No variation in X variable".to_string(),
            });
        }
        
        let slope = numerator / denominator;
        let intercept = y_mean - slope * x_mean;
        let coefficients = vec![intercept, slope];
        
        // Calculate predicted values and residuals
        let mut residuals = Vec::new();
        let mut ss_res = 0.0; // Residual sum of squares
        let mut ss_tot = 0.0; // Total sum of squares
        
        for i in 0..n {
            let predicted = intercept + slope * x[i];
            let residual = y[i] - predicted;
            residuals.push(residual);
            ss_res += residual * residual;
            ss_tot += (y[i] - y_mean) * (y[i] - y_mean);
        }
        
        // R-squared
        let r_squared = if ss_tot > 0.0 { 1.0 - ss_res / ss_tot } else { 0.0 };
        let adjusted_r_squared = 1.0 - (ss_res / (n - 2) as f64) / (ss_tot / (n - 1) as f64);
        
        // Standard errors and t-statistics
        let mse = ss_res / (n - 2) as f64; // Mean squared error
        let se_slope = (mse / denominator).sqrt();
        let se_intercept = (mse * (1.0 / n as f64 + x_mean * x_mean / denominator)).sqrt();
        
        let standard_errors = vec![se_intercept, se_slope];
        let t_statistics = vec![intercept / se_intercept, slope / se_slope];
        
        // P-values
        let df = n - 2;
        let p_values = vec![
            2.0 * (1.0 - self.t_cdf(t_statistics[0].abs(), df)),
            2.0 * (1.0 - self.t_cdf(t_statistics[1].abs(), df))
        ];
        
        // F-statistic for overall model
        let f_statistic = (r_squared / 1.0) / ((1.0 - r_squared) / (n - 2) as f64);

        Ok(RegressionResult {
            coefficients,
            r_squared,
            adjusted_r_squared,
            standard_errors,
            t_statistics,
            p_values,
            residual_ss: ss_res,
            total_ss: ss_tot,
            f_statistic,
            df_model: 1,
            df_residual: n - 2,
        })
    }

    /// Calculate confidence interval for mean
    pub fn confidence_interval_mean(&self, data: &[f64], confidence_level: f64) -> Result<ConfidenceInterval> {
        if data.len() < 2 {
            return Err(NEATError::InvalidConfiguration {
                parameter: "sample_size".to_string(),
                value: "Need at least 2 observations for confidence interval".to_string(),
            });
        }

        let n = data.len();
        let mean = self.mean(data);
        let std_dev = self.variance(data, mean).sqrt();
        let standard_error = std_dev / (n as f64).sqrt();
        
        let alpha = 1.0 - confidence_level;
        let df = n - 1;
        let t_critical = self.t_inverse(1.0 - alpha / 2.0, df);
        
        let margin_of_error = t_critical * standard_error;
        let lower_bound = mean - margin_of_error;
        let upper_bound = mean + margin_of_error;

        Ok(ConfidenceInterval {
            lower_bound,
            upper_bound,
            confidence_level,
            margin_of_error,
            statistic: "mean".to_string(),
        })
    }

    /// Calculate normal distribution probability
    pub fn normal_probability(&self, x: f64, mean: f64, std_dev: f64) -> Result<f64> {
        if std_dev <= 0.0 {
            return Err(NEATError::InvalidConfiguration {
                parameter: "standard_deviation".to_string(),
                value: "Standard deviation must be positive".to_string(),
            });
        }

        let z = (x - mean) / std_dev;
        let probability = (1.0 / (std_dev * (2.0 * PI).sqrt())) * (-0.5 * z * z).exp();
        Ok(probability)
    }

    /// Calculate normal cumulative distribution function
    pub fn normal_cdf(&self, x: f64, mean: f64, std_dev: f64) -> Result<f64> {
        if std_dev <= 0.0 {
            return Err(NEATError::InvalidConfiguration {
                parameter: "standard_deviation".to_string(),
                value: "Standard deviation must be positive".to_string(),
            });
        }

        let z = (x - mean) / std_dev;
        Ok(self.standard_normal_cdf(z))
    }

    // Helper methods

    /// Calculate arithmetic mean
    fn mean(&self, data: &[f64]) -> f64 {
        data.iter().sum::<f64>() / data.len() as f64
    }

    /// Calculate median
    fn median(&self, sorted_data: &[f64]) -> f64 {
        let n = sorted_data.len();
        if n % 2 == 0 {
            (sorted_data[n / 2 - 1] + sorted_data[n / 2]) / 2.0
        } else {
            sorted_data[n / 2]
        }
    }

    /// Calculate mode (most frequent value)
    fn mode(&self, data: &[f64]) -> Option<f64> {
        let mut frequency_map = HashMap::new();
        
        for &value in data {
            *frequency_map.entry(value.to_bits()).or_insert(0) += 1;
        }
        
        frequency_map.iter()
            .max_by_key(|(_, &count)| count)
            .map(|(&bits, _)| f64::from_bits(bits))
    }

    /// Calculate sample variance
    fn variance(&self, data: &[f64], mean: f64) -> f64 {
        let n = data.len();
        if n < 2 {
            return 0.0;
        }
        
        let sum_squared_diff: f64 = data.iter()
            .map(|&x| (x - mean).powi(2))
            .sum();
            
        sum_squared_diff / (n - 1) as f64
    }

    /// Calculate percentile
    fn percentile(&self, sorted_data: &[f64], percentile: f64) -> f64 {
        let n = sorted_data.len();
        let index = (percentile / 100.0) * (n - 1) as f64;
        let lower = index.floor() as usize;
        let upper = index.ceil() as usize;
        
        if lower == upper {
            sorted_data[lower]
        } else {
            let fraction = index - lower as f64;
            sorted_data[lower] * (1.0 - fraction) + sorted_data[upper] * fraction
        }
    }

    /// Calculate skewness
    fn skewness(&self, data: &[f64], mean: f64, std_dev: f64) -> f64 {
        let n = data.len() as f64;
        if std_dev == 0.0 {
            return 0.0;
        }
        
        let sum_cubed: f64 = data.iter()
            .map(|&x| ((x - mean) / std_dev).powi(3))
            .sum();
            
        (n / ((n - 1.0) * (n - 2.0))) * sum_cubed
    }

    /// Calculate kurtosis
    fn kurtosis(&self, data: &[f64], mean: f64, std_dev: f64) -> f64 {
        let n = data.len() as f64;
        if std_dev == 0.0 {
            return 0.0;
        }
        
        let sum_fourth: f64 = data.iter()
            .map(|&x| ((x - mean) / std_dev).powi(4))
            .sum();
            
        let numerator = n * (n + 1.0) * sum_fourth / ((n - 1.0) * (n - 2.0) * (n - 3.0));
        let adjustment = 3.0 * (n - 1.0).powi(2) / ((n - 2.0) * (n - 3.0));
        
        numerator - adjustment
    }

    /// Standard normal CDF approximation
    fn standard_normal_cdf(&self, z: f64) -> f64 {
        // Abramowitz and Stegun approximation
        if z < 0.0 {
            return 1.0 - self.standard_normal_cdf(-z);
        }
        
        let t = 1.0 / (1.0 + 0.2316419 * z);
        let a1 =  0.319381530;
        let a2 = -0.356563782;
        let a3 =  1.781477937;
        let a4 = -1.821255978;
        let a5 =  1.330274429;
        
        let polynomial = a1 * t + a2 * t.powi(2) + a3 * t.powi(3) + a4 * t.powi(4) + a5 * t.powi(5);
        let phi = 1.0 - ((-z * z / 2.0).exp() / (2.0 * PI).sqrt()) * polynomial;
        
        phi
    }

    /// T-distribution CDF approximation
    fn t_cdf(&self, t: f64, df: usize) -> f64 {
        // Approximation using normal distribution for large df
        if df >= 30 {
            return self.standard_normal_cdf(t);
        }
        
        // Simple approximation for small df
        let x = t / (df as f64).sqrt();
        let beta_half = 0.5;
        let beta_df_half = df as f64 / 2.0;
        
        // This is a simplified approximation
        0.5 + (x * (1.0 + x * x / df as f64).powf(-beta_df_half - 0.5)) / 2.0
    }

    /// T-distribution inverse CDF approximation
    fn t_inverse(&self, p: f64, df: usize) -> f64 {
        // For large df, use normal approximation
        if df >= 30 {
            return self.normal_inverse(p);
        }
        
        // Simple approximation for t-distribution
        let normal_quantile = self.normal_inverse(p);
        let correction = normal_quantile.powi(3) / (4.0 * df as f64);
        
        normal_quantile + correction
    }

    /// Normal distribution inverse CDF (quantile function)
    fn normal_inverse(&self, p: f64) -> f64 {
        // Beasley-Springer-Moro algorithm approximation
        if p <= 0.0 || p >= 1.0 {
            return if p <= 0.0 { f64::NEG_INFINITY } else { f64::INFINITY };
        }
        
        let a0 = -3.969683028665376e+01;
        let a1 =  2.209460984245205e+02;
        let a2 = -2.759285104469687e+02;
        let a3 =  1.383577518672690e+02;
        let a4 = -3.066479806614716e+01;
        let a5 =  2.506628277459239e+00;
        
        let b1 = -5.447609879822406e+01;
        let b2 =  1.615858368580409e+02;
        let b3 = -1.556989798598866e+02;
        let b4 =  6.680131188771972e+01;
        let b5 = -1.328068155288572e+01;
        
        let y = if p < 0.5 { p } else { 1.0 - p };
        let r = (-2.0 * y.ln()).sqrt();
        
        let num = a0 + a1 * r + a2 * r.powi(2) + a3 * r.powi(3) + a4 * r.powi(4) + a5 * r.powi(5);
        let den = 1.0 + b1 * r + b2 * r.powi(2) + b3 * r.powi(3) + b4 * r.powi(4) + b5 * r.powi(5);
        
        let x = r - num / den;
        
        if p < 0.5 { -x } else { x }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_descriptive_statistics() -> Result<()> {
        let engine = StatisticsEngine::default();
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        
        let stats = engine.descriptive_statistics(&data)?;
        
        assert_eq!(stats.n, 5);
        assert!((stats.mean - 3.0).abs() < 1e-10);
        assert!((stats.median - 3.0).abs() < 1e-10);
        assert_eq!(stats.min, 1.0);
        assert_eq!(stats.max, 5.0);
        assert!((stats.std_dev - (2.5_f64).sqrt()).abs() < 1e-10);
        
        Ok(())
    }

    #[test]
    fn test_one_sample_t_test() -> Result<()> {
        let engine = StatisticsEngine::default();
        let data = vec![2.1, 2.3, 1.9, 2.0, 2.2, 1.8, 2.4]; // Mean should be around 2.1
        
        let result = engine.one_sample_t_test(&data, 2.0, Some(0.05))?;
        
        assert!(result.test_statistic > 0.0); // Should be positive since sample mean > 2.0
        assert!(result.p_value >= 0.0 && result.p_value <= 1.0);
        assert_eq!(result.degrees_of_freedom, Some(6));
        assert!(result.effect_size.is_some());
        
        Ok(())
    }

    #[test]
    fn test_two_sample_t_test() -> Result<()> {
        let engine = StatisticsEngine::default();
        let data1 = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let data2 = vec![2.0, 3.0, 4.0, 5.0, 6.0];
        
        let result = engine.two_sample_t_test(&data1, &data2, Some(0.05))?;
        
        assert!(result.test_statistic < 0.0); // data1 mean < data2 mean
        assert!(result.p_value >= 0.0 && result.p_value <= 1.0);
        assert!(result.degrees_of_freedom.is_some());
        
        Ok(())
    }

    #[test]
    fn test_linear_regression() -> Result<()> {
        let engine = StatisticsEngine::default();
        let x = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let y = vec![2.0, 4.0, 6.0, 8.0, 10.0]; // Perfect linear relationship: y = 2x
        
        let result = engine.linear_regression(&x, &y)?;
        
        assert!((result.coefficients[0] - 0.0).abs() < 1e-10); // Intercept should be 0
        assert!((result.coefficients[1] - 2.0).abs() < 1e-10); // Slope should be 2
        assert!((result.r_squared - 1.0).abs() < 1e-10); // Perfect fit
        
        Ok(())
    }

    #[test]
    fn test_confidence_interval() -> Result<()> {
        let engine = StatisticsEngine::default();
        let data = vec![10.0, 12.0, 13.0, 15.0, 18.0, 20.0, 16.0, 14.0];
        
        let ci = engine.confidence_interval_mean(&data, 0.95)?;
        
        assert!(ci.lower_bound < ci.upper_bound);
        assert_eq!(ci.confidence_level, 0.95);
        assert!(ci.margin_of_error > 0.0);
        
        // The interval should contain the sample mean
        let mean = engine.mean(&data);
        assert!(ci.lower_bound <= mean && mean <= ci.upper_bound);
        
        Ok(())
    }

    #[test]
    fn test_normal_distribution() -> Result<()> {
        let engine = StatisticsEngine::default();
        
        // Test standard normal distribution
        let prob = engine.normal_probability(0.0, 0.0, 1.0)?;
        let expected = 1.0 / (2.0 * PI).sqrt();
        assert!((prob - expected).abs() < 1e-10);
        
        // Test normal CDF
        let cdf_0 = engine.normal_cdf(0.0, 0.0, 1.0)?;
        assert!((cdf_0 - 0.5).abs() < 1e-3);
        
        Ok(())
    }

    #[test]
    fn test_percentiles() -> Result<()> {
        let engine = StatisticsEngine::default();
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
        
        let stats = engine.descriptive_statistics(&data)?;
        
        assert!((stats.q1 - 3.25).abs() < 0.1); // 25th percentile
        assert!((stats.median - 5.5).abs() < 0.1); // 50th percentile
        assert!((stats.q3 - 7.75).abs() < 0.1); // 75th percentile
        
        Ok(())
    }

    #[test]
    fn test_variance_and_std_dev() -> Result<()> {
        let engine = StatisticsEngine::default();
        let data = vec![2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0];
        
        let stats = engine.descriptive_statistics(&data)?;
        
        // Verify variance calculation
        assert!(stats.variance > 0.0);
        assert!((stats.std_dev - stats.variance.sqrt()).abs() < 1e-10);
        
        Ok(())
    }

    #[test]
    fn test_empty_dataset_error() {
        let engine = StatisticsEngine::default();
        let empty_data: Vec<f64> = vec![];
        
        let result = engine.descriptive_statistics(&empty_data);
        assert!(result.is_err());
    }
}