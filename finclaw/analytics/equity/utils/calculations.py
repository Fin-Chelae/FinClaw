"""Equity Investment Calculations Module
======================================

Financial calculations and utility functions.
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Any, Optional, Tuple, Union
import math
from scipy import stats
from scipy.optimize import fsolve
import warnings

from ..base.base_models import ValidationError

TRADING_DAYS_PER_YEAR = 252


def _to_series(data: Union[List[float], pd.Series]) -> pd.Series:
    """Coerce list or Series input to pd.Series."""
    return pd.Series(data) if isinstance(data, list) else data


class FinancialCalculations:
    """Common financial calculation utilities"""

    @staticmethod
    def time_value_of_money(principal: float, rate: float, periods: int,
                            compounding: str = "annual") -> Dict[str, float]:
        """Comprehensive time value of money calculations"""

        compounding_freq = {
            "annual": 1,
            "semi-annual": 2,
            "quarterly": 4,
            "monthly": 12,
            "daily": 365,
            "continuous": float('inf')
        }

        freq = compounding_freq.get(compounding.lower(), 1)

        if freq == float('inf'):
            future_value = principal * math.exp(rate * periods)
            effective_rate = math.exp(rate) - 1
        else:
            future_value = principal * (1 + rate / freq) ** (freq * periods)
            effective_rate = (1 + rate / freq) ** freq - 1

        present_value = future_value / ((1 + effective_rate) ** periods)

        return {
            'principal': principal,
            'future_value': future_value,
            'present_value_of_fv': present_value,
            'effective_annual_rate': effective_rate,
            'total_interest': future_value - principal,
            'compounding_frequency': freq
        }

    @staticmethod
    def annuity_calculations(payment: float, rate: float, periods: int,
                             annuity_type: str = "ordinary") -> Dict[str, float]:
        """Calculate present and future value of annuities"""

        if rate <= 0:
            pv_annuity = payment * periods
            fv_annuity = payment * periods
        else:
            pv_ordinary = payment * ((1 - (1 + rate) ** -periods) / rate)
            fv_ordinary = payment * (((1 + rate) ** periods - 1) / rate)

            if annuity_type.lower() == "due":
                pv_annuity = pv_ordinary * (1 + rate)
                fv_annuity = fv_ordinary * (1 + rate)
            else:
                pv_annuity = pv_ordinary
                fv_annuity = fv_ordinary

        return {
            'payment_amount': payment,
            'present_value': pv_annuity,
            'future_value': fv_annuity,
            'total_payments': payment * periods,
            'total_interest': fv_annuity - (payment * periods),
            'annuity_type': annuity_type
        }

    @staticmethod
    def perpetuity_value(payment: float, discount_rate: float,
                         growth_rate: float = 0) -> Dict[str, float]:
        """Calculate present value of perpetuity"""

        if discount_rate <= growth_rate:
            raise ValidationError("Discount rate must be greater than growth rate")

        if growth_rate == 0:
            pv = payment / discount_rate
        else:
            pv = payment / (discount_rate - growth_rate)

        return {
            'payment': payment,
            'discount_rate': discount_rate,
            'growth_rate': growth_rate,
            'present_value': pv,
            'perpetuity_type': 'Growing' if growth_rate > 0 else 'Simple'
        }

    @staticmethod
    def loan_calculations(principal: float, annual_rate: float, years: int,
                          payment_frequency: int = 12) -> Dict[str, Any]:
        """Calculate loan payments and amortization"""

        monthly_rate = annual_rate / payment_frequency
        total_payments = years * payment_frequency

        if annual_rate == 0:
            payment = principal / total_payments
        else:
            payment = principal * (monthly_rate * (1 + monthly_rate) ** total_payments) / \
                      ((1 + monthly_rate) ** total_payments - 1)

        balance = principal
        schedule = []
        total_interest = 0

        for i in range(1, int(total_payments) + 1):
            interest_payment = balance * monthly_rate
            principal_payment = payment - interest_payment
            balance -= principal_payment
            total_interest += interest_payment

            schedule.append({
                'payment_number': i,
                'payment': payment,
                'principal': principal_payment,
                'interest': interest_payment,
                'balance': max(0, balance)
            })

        return {
            'loan_amount': principal,
            'monthly_payment': payment,
            'total_payments': total_payments,
            'total_interest': total_interest,
            'total_cost': principal + total_interest,
            'amortization_schedule': schedule[:12],
            'full_schedule_available': True
        }

    @staticmethod
    def bond_calculations(face_value: float, coupon_rate: float, market_rate: float,
                          years_to_maturity: float, frequency: int = 2) -> Dict[str, float]:
        """Calculate bond price, yield, and duration"""

        periods = years_to_maturity * frequency
        coupon_payment = (face_value * coupon_rate) / frequency
        period_rate = market_rate / frequency

        if market_rate == 0:
            bond_price = face_value + (coupon_payment * periods)
        else:
            pv_coupons = coupon_payment * ((1 - (1 + period_rate) ** -periods) / period_rate)
            pv_face = face_value / ((1 + period_rate) ** periods)
            bond_price = pv_coupons + pv_face

        current_yield = (coupon_payment * frequency) / bond_price

        cash_flows = [coupon_payment] * int(periods)
        cash_flows[-1] += face_value

        weighted_time = 0
        total_pv = 0

        for t, cf in enumerate(cash_flows, 1):
            pv_cf = cf / ((1 + period_rate) ** t)
            weighted_time += (t / frequency) * pv_cf
            total_pv += pv_cf

        macaulay_duration = weighted_time / total_pv
        modified_duration = macaulay_duration / (1 + market_rate / frequency)

        return {
            'bond_price': bond_price,
            'face_value': face_value,
            'coupon_rate': coupon_rate,
            'market_rate': market_rate,
            'current_yield': current_yield,
            'macaulay_duration': macaulay_duration,
            'modified_duration': modified_duration,
            'price_sensitivity': modified_duration * bond_price * 0.01,
            'premium_discount': 'Premium' if bond_price > face_value else 'Discount' if bond_price < face_value else 'Par'
        }


class StatisticalCalculations:
    """Statistical analysis utilities for finance"""

    @staticmethod
    def descriptive_statistics(data: Union[List[float], pd.Series]) -> Dict[str, float]:
        """Calculate comprehensive descriptive statistics"""

        if isinstance(data, list):
            data = pd.Series(data)

        return {
            'count': len(data),
            'mean': data.mean(),
            'median': data.median(),
            'mode': data.mode().iloc[0] if not data.mode().empty else np.nan,
            'std_dev': data.std(),
            'variance': data.var(),
            'skewness': data.skew(),
            'kurtosis': data.kurtosis(),
            'min': data.min(),
            'max': data.max(),
            'range': data.max() - data.min(),
            'q25': data.quantile(0.25),
            'q75': data.quantile(0.75),
            'iqr': data.quantile(0.75) - data.quantile(0.25),
            'cv': data.std() / data.mean() if data.mean() != 0 else np.nan
        }

    @staticmethod
    def correlation_analysis(x: Union[List[float], pd.Series],
                             y: Union[List[float], pd.Series]) -> Dict[str, float]:
        """Calculate correlation and regression statistics"""

        if isinstance(x, list):
            x = pd.Series(x)
        if isinstance(y, list):
            y = pd.Series(y)

        valid_data = pd.DataFrame({'x': x, 'y': y}).dropna()
        x_clean = valid_data['x']
        y_clean = valid_data['y']

        if len(x_clean) < 2:
            return {'error': 'Insufficient data for correlation analysis'}

        pearson_corr = x_clean.corr(y_clean)
        spearman_corr = x_clean.corr(y_clean, method='spearman')

        slope, intercept, r_value, p_value, std_err = stats.linregress(x_clean, y_clean)

        return {
            'pearson_correlation': pearson_corr,
            'spearman_correlation': spearman_corr,
            'r_squared': r_value ** 2,
            'regression_slope': slope,
            'regression_intercept': intercept,
            'p_value': p_value,
            'standard_error': std_err,
            'sample_size': len(x_clean)
        }

    @staticmethod
    def hypothesis_testing(sample_data: Union[List[float], pd.Series],
                           null_hypothesis: float, alternative: str = "two-sided",
                           alpha: float = 0.05) -> Dict[str, Any]:
        """Perform one-sample t-test"""

        if isinstance(sample_data, list):
            sample_data = pd.Series(sample_data)

        clean_data = sample_data.dropna()

        if len(clean_data) < 2:
            return {'error': 'Insufficient data for hypothesis testing'}

        t_stat, p_value = stats.ttest_1samp(clean_data, null_hypothesis)

        if alternative == "two-sided":
            reject_null = p_value < alpha
        elif alternative == "greater":
            reject_null = (t_stat > 0) and (p_value / 2 < alpha)
        elif alternative == "less":
            reject_null = (t_stat < 0) and (p_value / 2 < alpha)
        else:
            reject_null = p_value < alpha

        confidence_level = 1 - alpha
        margin_error = stats.t.ppf((1 + confidence_level) / 2, len(clean_data) - 1) * \
                       (clean_data.std() / math.sqrt(len(clean_data)))

        ci_lower = clean_data.mean() - margin_error
        ci_upper = clean_data.mean() + margin_error

        return {
            'sample_mean': clean_data.mean(),
            'null_hypothesis': null_hypothesis,
            't_statistic': t_stat,
            'p_value': p_value,
            'alpha': alpha,
            'reject_null': reject_null,
            'confidence_interval': (ci_lower, ci_upper),
            'confidence_level': confidence_level,
            'sample_size': len(clean_data),
            'degrees_freedom': len(clean_data) - 1
        }


class RiskMetrics:
    """Risk and return calculation utilities"""

    @staticmethod
    def portfolio_metrics(returns: Union[List[float], pd.Series],
                          risk_free_rate: float = 0.02) -> Dict[str, float]:
        """Calculate comprehensive portfolio risk metrics"""

        if isinstance(returns, list):
            returns = pd.Series(returns)

        returns = returns.dropna()

        if len(returns) == 0:
            return {'error': 'No valid return data'}

        mean_return = returns.mean()
        volatility = returns.std()

        daily_rf = risk_free_rate / TRADING_DAYS_PER_YEAR
        sharpe_ratio = (mean_return - daily_rf) / volatility if volatility > 0 else 0

        downside_returns = returns[returns < 0]
        downside_deviation = downside_returns.std() if len(downside_returns) > 0 else 0
        sortino_ratio = (mean_return - daily_rf) / downside_deviation if downside_deviation > 0 else 0

        var_95 = np.percentile(returns, 5)
        var_99 = np.percentile(returns, 1)

        es_95 = returns[returns <= var_95].mean() if len(returns[returns <= var_95]) > 0 else var_95
        es_99 = returns[returns <= var_99].mean() if len(returns[returns <= var_99]) > 0 else var_99

        cumulative_returns = (1 + returns).cumprod()
        running_max = cumulative_returns.expanding().max()
        drawdown = (cumulative_returns - running_max) / running_max
        max_drawdown = drawdown.min()

        return {
            'mean_return': mean_return,
            'annualized_return': mean_return * TRADING_DAYS_PER_YEAR,
            'volatility': volatility,
            'annualized_volatility': volatility * math.sqrt(TRADING_DAYS_PER_YEAR),
            'sharpe_ratio': sharpe_ratio,
            'sortino_ratio': sortino_ratio,
            'var_95': var_95,
            'var_99': var_99,
            'expected_shortfall_95': es_95,
            'expected_shortfall_99': es_99,
            'max_drawdown': max_drawdown,
            'downside_deviation': downside_deviation,
            'positive_periods': (returns > 0).sum(),
            'negative_periods': (returns < 0).sum(),
            'hit_ratio': (returns > 0).sum() / len(returns)
        }

    @staticmethod
    def beta_calculation(asset_returns: Union[List[float], pd.Series],
                         market_returns: Union[List[float], pd.Series],
                         risk_free_rate: float = 0.02) -> Dict[str, float]:
        """Calculate beta and related risk metrics"""

        if isinstance(asset_returns, list):
            asset_returns = pd.Series(asset_returns)
        if isinstance(market_returns, list):
            market_returns = pd.Series(market_returns)

        combined = pd.DataFrame({'asset': asset_returns, 'market': market_returns}).dropna()

        if len(combined) < 10:
            return {'error': 'Insufficient data for beta calculation'}

        asset_clean = combined['asset']
        market_clean = combined['market']

        covariance = asset_clean.cov(market_clean)
        market_variance = market_clean.var()
        beta = covariance / market_variance if market_variance > 0 else 0

        daily_rf = risk_free_rate / TRADING_DAYS_PER_YEAR
        alpha = asset_clean.mean() - (daily_rf + beta * (market_clean.mean() - daily_rf))

        correlation = asset_clean.corr(market_clean)
        r_squared = correlation ** 2

        excess_returns = asset_clean - market_clean
        tracking_error = excess_returns.std()

        information_ratio = excess_returns.mean() / tracking_error if tracking_error > 0 else 0

        return {
            'beta': beta,
            'alpha': alpha,
            'correlation': correlation,
            'r_squared': r_squared,
            'tracking_error': tracking_error,
            'information_ratio': information_ratio,
            'sample_size': len(combined),
            'annualized_alpha': alpha * TRADING_DAYS_PER_YEAR,
            'annualized_tracking_error': tracking_error * math.sqrt(TRADING_DAYS_PER_YEAR)
        }


class OptionCalculations:
    """Option pricing and Greeks calculations"""

    @staticmethod
    def black_scholes(spot_price: float, strike_price: float, time_to_expiry: float,
                      risk_free_rate: float, volatility: float,
                      option_type: str = "call") -> Dict[str, float]:
        """Calculate Black-Scholes option price and Greeks"""

        d1 = (math.log(spot_price / strike_price) +
              (risk_free_rate + 0.5 * volatility ** 2) * time_to_expiry) / \
             (volatility * math.sqrt(time_to_expiry))

        d2 = d1 - volatility * math.sqrt(time_to_expiry)

        N_d1 = stats.norm.cdf(d1)
        N_d2 = stats.norm.cdf(d2)
        N_neg_d1 = stats.norm.cdf(-d1)
        N_neg_d2 = stats.norm.cdf(-d2)

        n_d1 = stats.norm.pdf(d1)

        if option_type.lower() == "call":
            option_price = spot_price * N_d1 - strike_price * math.exp(-risk_free_rate * time_to_expiry) * N_d2
            delta = N_d1
            gamma = n_d1 / (spot_price * volatility * math.sqrt(time_to_expiry))
            theta = (-spot_price * n_d1 * volatility / (2 * math.sqrt(time_to_expiry)) -
                     risk_free_rate * strike_price * math.exp(-risk_free_rate * time_to_expiry) * N_d2) / 365

        else:
            option_price = strike_price * math.exp(-risk_free_rate * time_to_expiry) * N_neg_d2 - spot_price * N_neg_d1
            delta = N_d1 - 1
            gamma = n_d1 / (spot_price * volatility * math.sqrt(time_to_expiry))
            theta = (-spot_price * n_d1 * volatility / (2 * math.sqrt(time_to_expiry)) +
                     risk_free_rate * strike_price * math.exp(-risk_free_rate * time_to_expiry) * N_neg_d2) / 365

        vega = spot_price * n_d1 * math.sqrt(time_to_expiry) / 100
        rho = (strike_price * time_to_expiry * math.exp(-risk_free_rate * time_to_expiry) *
               (N_d2 if option_type.lower() == "call" else N_neg_d2)) / 100

        return {
            'option_price': option_price,
            'delta': delta,
            'gamma': gamma,
            'theta': theta,
            'vega': vega,
            'rho': rho,
            'd1': d1,
            'd2': d2,
            'intrinsic_value': max(0, spot_price - strike_price) if option_type.lower() == "call"
            else max(0, strike_price - spot_price),
            'time_value': option_price - max(0, spot_price - strike_price if option_type.lower() == "call"
            else strike_price - spot_price)
        }

    @staticmethod
    def implied_volatility(option_price: float, spot_price: float, strike_price: float,
                           time_to_expiry: float, risk_free_rate: float,
                           option_type: str = "call") -> float:
        """Calculate implied volatility using Newton-Raphson method"""

        def bs_price_diff(vol):
            bs_result = OptionCalculations.black_scholes(
                spot_price, strike_price, time_to_expiry, risk_free_rate, vol, option_type
            )
            return bs_result['option_price'] - option_price

        try:
            initial_vol = 0.2
            implied_vol = fsolve(bs_price_diff, initial_vol)[0]

            if implied_vol < 0 or implied_vol > 5:
                return np.nan

            return implied_vol

        except Exception:
            return np.nan


class TechnicalIndicators:
    """Technical analysis calculation utilities"""

    @staticmethod
    def moving_averages(prices: Union[List[float], pd.Series],
                        periods: List[int]) -> Dict[str, pd.Series]:
        """Calculate multiple moving averages"""

        prices = _to_series(prices)
        moving_avgs = {}

        for period in periods:
            ma_name = f'MA_{period}'
            moving_avgs[ma_name] = prices.rolling(window=period).mean()

        return moving_avgs

    @staticmethod
    def bollinger_bands(prices: Union[List[float], pd.Series],
                        period: int = 20, std_dev: float = 2) -> Dict[str, pd.Series]:
        """Calculate Bollinger Bands"""

        prices = _to_series(prices)
        ma = prices.rolling(window=period).mean()
        std = prices.rolling(window=period).std()

        upper_band = ma + (std * std_dev)
        lower_band = ma - (std * std_dev)

        return {
            'middle_band': ma,
            'upper_band': upper_band,
            'lower_band': lower_band,
            'bandwidth': (upper_band - lower_band) / ma,
            'percent_b': (prices - lower_band) / (upper_band - lower_band)
        }

    @staticmethod
    def rsi(prices: Union[List[float], pd.Series], period: int = 14) -> pd.Series:
        """Calculate Relative Strength Index"""

        prices = _to_series(prices)
        delta = prices.diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)

        avg_gain = gain.rolling(window=period).mean()
        avg_loss = loss.rolling(window=period).mean()

        rs = avg_gain / avg_loss
        rsi_values = 100 - (100 / (1 + rs))

        return rsi_values

    @staticmethod
    def macd(prices: Union[List[float], pd.Series],
             fast_period: int = 12, slow_period: int = 26, signal_period: int = 9) -> Dict[str, pd.Series]:
        """Calculate MACD indicator"""

        prices = _to_series(prices)
        ema_fast = prices.ewm(span=fast_period).mean()
        ema_slow = prices.ewm(span=slow_period).mean()

        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=signal_period).mean()
        histogram = macd_line - signal_line

        return {
            'macd_line': macd_line,
            'signal_line': signal_line,
            'histogram': histogram
        }


# Utility functions for common calculations
def quick_return_calculation(start_price: float, end_price: float,
                             dividends: float = 0) -> Dict[str, float]:
    """Quick return calculation"""
    price_return = (end_price - start_price) / start_price
    total_return = (end_price - start_price + dividends) / start_price

    return {
        'price_return': price_return,
        'total_return': total_return,
        'dividend_yield': dividends / start_price,
        'price_appreciation': price_return
    }


def compound_annual_growth_rate(beginning_value: float, ending_value: float,
                                years: float) -> float:
    """Calculate CAGR"""
    if beginning_value <= 0 or ending_value <= 0 or years <= 0:
        raise ValidationError("All values must be positive for CAGR calculation")

    return (ending_value / beginning_value) ** (1 / years) - 1


def rule_of_72(interest_rate: float) -> float:
    """Calculate doubling time using Rule of 72"""
    if interest_rate <= 0:
        raise ValidationError("Interest rate must be positive")

    return 72 / (interest_rate * 100)


def effective_annual_rate(nominal_rate: float, compounding_periods: int) -> float:
    """Calculate effective annual rate"""
    return (1 + nominal_rate / compounding_periods) ** compounding_periods - 1


def present_value_growing_annuity(payment: float, growth_rate: float,
                                  discount_rate: float, periods: int) -> float:
    """Calculate PV of growing annuity"""
    if discount_rate == growth_rate:
        return payment * periods / (1 + discount_rate)

    factor = (1 - ((1 + growth_rate) / (1 + discount_rate)) ** periods)
    return payment * factor / (discount_rate - growth_rate)
