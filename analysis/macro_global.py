"""
Global macro-economic regime detection using the World Bank API.

Analyzes macroeconomic conditions for 200+ countries to adjust position
sizing and risk exposure based on GDP growth, inflation, real interest
rates, unemployment, and current account balance.

This is the international equivalent of the US-only FRED-based macro.py.

Requires: requests library (no API key needed)
  World Bank API docs: https://datahelpdesk.worldbank.org/knowledgebase/articles/889392
"""

import logging
import time
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

import requests

from core.config import REGIME_POSITION_MODIFIERS

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# World Bank indicator codes
# ---------------------------------------------------------------------------
INDICATORS: Dict[str, Dict[str, str]] = {
    "gdp_growth": {
        "code": "NY.GDP.MKTP.KD.ZG",
        "name": "GDP Growth Rate",
        "unit": "% annual",
    },
    "inflation": {
        "code": "FP.CPI.TOTL.ZG",
        "name": "Inflation (CPI)",
        "unit": "% annual",
    },
    "real_interest_rate": {
        "code": "FR.INR.RINR",
        "name": "Real Interest Rate",
        "unit": "%",
    },
    "unemployment": {
        "code": "SL.UEM.TOTL.ZS",
        "name": "Unemployment Rate",
        "unit": "% of labor force",
    },
    "current_account": {
        "code": "BN.CAB.XOKA.GD.ZS",
        "name": "Current Account Balance",
        "unit": "% of GDP",
    },
}

# World Bank API base URL
WB_API_BASE = "http://api.worldbank.org/v2"

# Default date range for queries (recent years)
_CURRENT_YEAR = datetime.now().year
DEFAULT_DATE_RANGE = f"{_CURRENT_YEAR - 6}:{_CURRENT_YEAR}"

# Request timeout in seconds
REQUEST_TIMEOUT = 15

# Cache TTL
CACHE_TTL_HOURS = 24


class _IndicatorCache:
    """Simple in-memory cache with per-key TTL."""

    def __init__(self, ttl: timedelta) -> None:
        self._store: Dict[str, Tuple[float, Any]] = {}
        self._ttl = ttl

    def get(self, key: str) -> Optional[Any]:
        entry = self._store.get(key)
        if entry is None:
            return None
        ts, value = entry
        if time.monotonic() - ts > self._ttl.total_seconds():
            del self._store[key]
            return None
        return value

    def set(self, key: str, value: Any) -> None:
        self._store[key] = (time.monotonic(), value)

    def clear(self) -> None:
        self._store.clear()


# Module-level cache shared across instances for the same process
_cache = _IndicatorCache(ttl=timedelta(hours=CACHE_TTL_HOURS))


class GlobalMacroAgent:
    """
    Global Market Regime Detector using World Bank macroeconomic data.

    Determines overall macroeconomic conditions for any country and
    provides risk modifiers to adjust position sizing.

    Risk Modifiers:
    - 1.0 = Normal conditions, full position sizes
    - 0.75 = Caution, reduce exposure
    - 0.5 = Elevated risk, half positions
    - 0.25 = High risk, minimal exposure
    - 0.0 = Critical, cash only
    """

    # Regime constants
    REGIME_BULLISH = "BULLISH"
    REGIME_NEUTRAL = "NEUTRAL"
    REGIME_CAUTIOUS = "CAUTIOUS"
    REGIME_BEARISH = "BEARISH"
    REGIME_CRITICAL = "CRITICAL"

    # --- GDP growth thresholds (annual %) ---
    GDP_STRONG = 3.0
    GDP_MODERATE = 1.5
    GDP_WEAK = 0.0
    GDP_RECESSION = -2.0

    # --- Inflation thresholds (annual %) ---
    INFLATION_LOW = 1.0
    INFLATION_MODERATE = 4.0
    INFLATION_HIGH = 8.0
    INFLATION_HYPER = 25.0

    # --- Unemployment thresholds (%) ---
    UNEMP_LOW = 4.0
    UNEMP_MODERATE = 6.0
    UNEMP_HIGH = 9.0
    UNEMP_CRISIS = 15.0

    # --- Real interest rate thresholds (%) ---
    REAL_RATE_VERY_NEGATIVE = -3.0
    REAL_RATE_NEGATIVE = 0.0
    REAL_RATE_HIGH = 5.0

    # --- Current account thresholds (% of GDP) ---
    CA_LARGE_DEFICIT = -6.0
    CA_DEFICIT = -3.0

    def __init__(self, country_code: str = "US") -> None:
        """
        Initialize GlobalMacroAgent.

        Args:
            country_code: ISO 3166-1 alpha-2 country code (e.g. "US", "GB",
                          "DE", "JP", "BR", "IN"). Defaults to "US".
        """
        self.country_code: str = country_code.upper()
        self._session: Optional[requests.Session] = None
        logger.info(
            "GlobalMacroAgent initialized for country=%s", self.country_code
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_session(self) -> requests.Session:
        """Lazily create a requests session with retry logic."""
        if self._session is None:
            self._session = requests.Session()
            adapter = requests.adapters.HTTPAdapter(
                max_retries=requests.adapters.Retry(
                    total=2,
                    backoff_factor=0.5,
                    status_forcelist=[429, 500, 502, 503, 504],
                ),
            )
            self._session.mount("http://", adapter)
            self._session.mount("https://", adapter)
        return self._session

    def _fetch_indicator(
        self,
        indicator_code: str,
        fallback: Optional[float] = None,
    ) -> Optional[float]:
        """
        Fetch the most recent non-null value for a World Bank indicator.

        Args:
            indicator_code: World Bank indicator code.
            fallback: Value to return if the fetch fails.

        Returns:
            Most recent value or *fallback*.
        """
        cache_key = f"{self.country_code}:{indicator_code}"
        cached = _cache.get(cache_key)
        if cached is not None:
            return cached

        url = (
            f"{WB_API_BASE}/country/{self.country_code}"
            f"/indicator/{indicator_code}"
        )
        params = {
            "format": "json",
            "date": DEFAULT_DATE_RANGE,
            "per_page": 50,
        }

        try:
            resp = self._get_session().get(
                url, params=params, timeout=REQUEST_TIMEOUT
            )
            resp.raise_for_status()
            payload = resp.json()

            # World Bank JSON responses: first element is paging metadata,
            # second element is the data array (or None if no data).
            if (
                not isinstance(payload, list)
                or len(payload) < 2
                or payload[1] is None
            ):
                logger.warning(
                    "No data returned for %s / %s",
                    self.country_code,
                    indicator_code,
                )
                return fallback

            # Iterate from most recent to oldest, return first non-null value
            for record in payload[1]:
                value = record.get("value")
                if value is not None:
                    value = float(value)
                    _cache.set(cache_key, value)
                    return value

            logger.warning(
                "All values null for %s / %s",
                self.country_code,
                indicator_code,
            )
            return fallback

        except requests.RequestException as exc:
            logger.warning(
                "World Bank API request failed for %s / %s: %s",
                self.country_code,
                indicator_code,
                exc,
            )
            return fallback
        except (ValueError, KeyError, IndexError, TypeError) as exc:
            logger.warning(
                "Failed to parse World Bank response for %s / %s: %s",
                self.country_code,
                indicator_code,
                exc,
            )
            return fallback

    def _fetch_all_indicators(self) -> Dict[str, Optional[float]]:
        """Fetch all configured indicators for the current country."""
        results: Dict[str, Optional[float]] = {}
        for key, meta in INDICATORS.items():
            results[key] = self._fetch_indicator(meta["code"])
        return results

    # ------------------------------------------------------------------
    # Individual indicator analysis
    # ------------------------------------------------------------------

    def _analyse_gdp(
        self, value: Optional[float]
    ) -> Tuple[Optional[float], str, float]:
        """Return (value, interpretation, risk_contribution)."""
        if value is None:
            return None, "Data unavailable", 0.0

        value = round(value, 2)
        if value >= self.GDP_STRONG:
            return value, "STRONG - Robust economic expansion", -0.5
        if value >= self.GDP_MODERATE:
            return value, "MODERATE - Steady growth", 0.0
        if value >= self.GDP_WEAK:
            return value, "WEAK - Growth stalling", 1.0
        if value >= self.GDP_RECESSION:
            return value, "CONTRACTION - Economy shrinking", 2.5
        return value, "DEEP RECESSION - Severe economic contraction", 3.5

    def _analyse_inflation(
        self, value: Optional[float]
    ) -> Tuple[Optional[float], str, float]:
        if value is None:
            return None, "Data unavailable", 0.0

        value = round(value, 2)
        if value < self.INFLATION_LOW:
            return value, "DEFLATION RISK - Prices stagnant or falling", 0.5
        if value <= self.INFLATION_MODERATE:
            return value, "STABLE - Inflation within target range", -0.5
        if value <= self.INFLATION_HIGH:
            return value, "ELEVATED - Inflation above comfort zone", 1.0
        if value <= self.INFLATION_HYPER:
            return value, "HIGH - Significant inflationary pressure", 2.5
        return value, "HYPERINFLATION - Currency losing value rapidly", 3.5

    def _analyse_unemployment(
        self, value: Optional[float]
    ) -> Tuple[Optional[float], str, float]:
        if value is None:
            return None, "Data unavailable", 0.0

        value = round(value, 2)
        if value < self.UNEMP_LOW:
            return value, "STRONG - Tight labor market", -0.5
        if value < self.UNEMP_MODERATE:
            return value, "HEALTHY - Normal employment levels", 0.0
        if value < self.UNEMP_HIGH:
            return value, "SOFTENING - Labor market weakening", 1.0
        if value < self.UNEMP_CRISIS:
            return value, "WEAK - Elevated unemployment", 2.0
        return value, "CRISIS - Mass unemployment", 3.0

    def _analyse_real_interest_rate(
        self, value: Optional[float]
    ) -> Tuple[Optional[float], str, float]:
        if value is None:
            return None, "Data unavailable", 0.0

        value = round(value, 2)
        if value < self.REAL_RATE_VERY_NEGATIVE:
            return (
                value,
                "DEEPLY NEGATIVE - Financial repression / high inflation",
                1.5,
            )
        if value < self.REAL_RATE_NEGATIVE:
            return value, "NEGATIVE - Accommodative real conditions", 0.5
        if value < self.REAL_RATE_HIGH:
            return value, "POSITIVE - Normal monetary conditions", 0.0
        return value, "RESTRICTIVE - Tight real monetary policy", 1.0

    def _analyse_current_account(
        self, value: Optional[float]
    ) -> Tuple[Optional[float], str, float]:
        if value is None:
            return None, "Data unavailable", 0.0

        value = round(value, 2)
        if value < self.CA_LARGE_DEFICIT:
            return (
                value,
                "LARGE DEFICIT - External vulnerability",
                1.5,
            )
        if value < self.CA_DEFICIT:
            return value, "DEFICIT - Moderate external imbalance", 0.5
        if value < 3.0:
            return value, "BALANCED - Healthy external position", 0.0
        return value, "SURPLUS - Strong external position", -0.5

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def get_market_regime(self) -> Dict:
        """
        Determine overall macro regime based on all indicators.

        Returns:
            Dict with regime, risk_score, risk_modifier, indicators,
            warnings, positives, recommendation, and timestamp.
        """
        raw = self._fetch_all_indicators()

        # Analyse each indicator
        analyses = {
            "gdp_growth": self._analyse_gdp(raw["gdp_growth"]),
            "inflation": self._analyse_inflation(raw["inflation"]),
            "unemployment": self._analyse_unemployment(raw["unemployment"]),
            "real_interest_rate": self._analyse_real_interest_rate(
                raw["real_interest_rate"]
            ),
            "current_account": self._analyse_current_account(
                raw["current_account"]
            ),
        }

        warnings: List[str] = []
        positives: List[str] = []
        risk_score: float = 0.0

        for key, (value, interp, contribution) in analyses.items():
            risk_score += contribution
            display = INDICATORS[key]["name"]
            if value is None:
                continue
            if contribution >= 1.5:
                warnings.append(f"{display}: {interp} ({value})")
            elif contribution >= 0.5:
                warnings.append(f"{display}: {interp} ({value})")
            elif contribution <= -0.25:
                positives.append(f"{display}: {interp} ({value})")

        # Clamp risk_score to a reasonable range for regime mapping
        risk_score = max(risk_score, -2.0)

        # Determine regime
        if risk_score >= 8:
            regime = self.REGIME_CRITICAL
            risk_modifier = REGIME_POSITION_MODIFIERS.get("CRITICAL", 0.0)
            recommendation = (
                "CASH ONLY - Severe macro deterioration across multiple fronts"
            )
        elif risk_score >= 5:
            regime = self.REGIME_BEARISH
            risk_modifier = REGIME_POSITION_MODIFIERS.get("BEARISH", 0.25)
            recommendation = (
                "MINIMAL EXPOSURE - Significant macro headwinds"
            )
        elif risk_score >= 3:
            regime = self.REGIME_CAUTIOUS
            risk_modifier = REGIME_POSITION_MODIFIERS.get("CAUTIOUS", 0.5)
            recommendation = "REDUCED POSITIONS - Elevated macro risk"
        elif risk_score >= 1.0:
            regime = self.REGIME_NEUTRAL
            risk_modifier = REGIME_POSITION_MODIFIERS.get("NEUTRAL", 0.75)
            recommendation = "MODERATE CAUTION - Mixed macro signals"
        else:
            regime = self.REGIME_BULLISH
            risk_modifier = REGIME_POSITION_MODIFIERS.get("BULLISH", 1.0)
            recommendation = (
                "FULL POSITIONS - Macro conditions broadly supportive"
            )

        # Build indicator detail dict (mirrors macro.py format)
        indicator_details: Dict[str, Dict] = {}
        for key, (value, interp, _) in analyses.items():
            indicator_details[key] = {
                "value": value,
                "interpretation": interp,
            }

        return {
            "regime": regime,
            "risk_modifier": risk_modifier,
            "risk_score": round(risk_score, 2),
            "recommendation": recommendation,
            "country": self.country_code,
            "timestamp": datetime.now().isoformat(),
            "indicators": indicator_details,
            "warnings": warnings,
            "positives": positives,
        }

    def get_position_size_modifier(self) -> float:
        """
        Get the risk modifier for position sizing.

        Returns:
            Float between 0.0 and 1.0 to multiply position sizes by.
        """
        regime_data = self.get_market_regime()
        return regime_data["risk_modifier"]

    def format_report(self) -> str:
        """
        Generate a formatted text report of current macro conditions.

        Returns:
            Formatted string report.
        """
        data = self.get_market_regime()

        lines = [
            "=" * 64,
            f"GLOBAL MACRO REGIME REPORT  --  {data['country']}",
            "=" * 64,
            f"Timestamp: {data['timestamp']}",
            "",
            f"REGIME:         {data['regime']}",
            f"RISK MODIFIER:  {data['risk_modifier']}"
            "  (multiply position sizes by this)",
            f"RISK SCORE:     {data['risk_score']}"
            "  (lower=safe, higher=critical)",
            "",
            f"RECOMMENDATION: {data['recommendation']}",
            "",
            "-" * 64,
            "INDICATORS:",
            "-" * 64,
        ]

        for key, info in data["indicators"].items():
            display_name = INDICATORS.get(key, {}).get(
                "name", key.replace("_", " ").title()
            )
            value = info["value"] if info["value"] is not None else "N/A"
            lines.append(f"  {display_name}: {value}")
            lines.append(f"    -> {info['interpretation']}")

        if data["warnings"]:
            lines.extend(
                [
                    "",
                    "-" * 64,
                    "WARNING SIGNALS:",
                    "-" * 64,
                ]
            )
            for warning in data["warnings"]:
                lines.append(f"  ! {warning}")

        if data["positives"]:
            lines.extend(
                [
                    "",
                    "-" * 64,
                    "POSITIVE SIGNALS:",
                    "-" * 64,
                ]
            )
            for positive in data["positives"]:
                lines.append(f"  + {positive}")

        lines.append("=" * 64)
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Convenience functions
# ---------------------------------------------------------------------------

def get_available_countries() -> List[Dict[str, str]]:
    """
    Return a list of countries supported by the World Bank API.

    Each entry is a dict with keys ``id`` (ISO alpha-2 code) and ``name``.
    Results are cached for 24 hours.

    Returns:
        List of dicts, e.g. [{"id": "US", "name": "United States"}, ...].
        Returns an empty list if the API is unreachable.
    """
    cache_key = "__countries__"
    cached = _cache.get(cache_key)
    if cached is not None:
        return cached

    url = f"{WB_API_BASE}/country"
    params = {"format": "json", "per_page": 400}
    countries: List[Dict[str, str]] = []

    try:
        resp = requests.get(url, params=params, timeout=REQUEST_TIMEOUT)
        resp.raise_for_status()
        payload = resp.json()

        if isinstance(payload, list) and len(payload) >= 2 and payload[1]:
            for entry in payload[1]:
                # Skip aggregate/regional codes (they have region id "NA")
                region = entry.get("region", {})
                if region.get("id") == "NA":
                    continue
                countries.append(
                    {
                        "id": entry.get("id", ""),
                        "name": entry.get("name", ""),
                    }
                )

        countries.sort(key=lambda c: c["name"])
        _cache.set(cache_key, countries)
        logger.info("Loaded %d country codes from World Bank API", len(countries))

    except requests.RequestException as exc:
        logger.warning("Failed to fetch country list from World Bank: %s", exc)
    except (ValueError, KeyError, TypeError) as exc:
        logger.warning("Failed to parse country list response: %s", exc)

    return countries


def get_global_macro_regime(country_code: str = "US") -> Dict:
    """
    Convenience function to get market regime without manually
    instantiating the class.

    Args:
        country_code: ISO 3166-1 alpha-2 country code.

    Returns:
        Dict with regime info or error message.
    """
    try:
        agent = GlobalMacroAgent(country_code=country_code)
        return agent.get_market_regime()
    except Exception as exc:
        logger.exception("Global macro analysis failed for %s", country_code)
        return {
            "error": str(exc),
            "regime": "UNKNOWN",
            "risk_modifier": 0.75,
            "country": country_code,
            "recommendation": (
                f"Macro analysis failed for {country_code}: {exc}"
            ),
        }
