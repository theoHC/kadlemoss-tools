#!/usr/bin/env python3
"""
Imaginary oceanic-climate hourly forecast generator.

- Mild temperatures year-round
- Small daily temperature swing
- Frequent clouds, drizzle, and occasional rain
- Coastal winds that can ramp up with rain events

Usage:
  python oceanic_hourly_forecast.py 2026 2 16
  python oceanic_hourly_forecast.py 2026-02-16
"""

from __future__ import annotations
import argparse
import math
import random
from dataclasses import dataclass
from datetime import date, datetime, timedelta


# -----------------------------
# Helpers
# -----------------------------

def clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))

def lerp(a: float, b: float, t: float) -> float:
    return a + (b - a) * t

def day_of_year(d: date) -> int:
    return int(d.strftime("%j"))

def fmt_wdir(deg: float) -> str:
    dirs = ["N","NNE","NE","ENE","E","ESE","SE","SSE","S","SSW","SW","WSW","W","WNW","NW","NNW"]
    idx = int((deg % 360) / 22.5 + 0.5) % 16
    return dirs[idx]

def choose_weighted(rng: random.Random, items):
    """items = [(value, weight), ...]"""
    total = sum(w for _, w in items)
    r = rng.random() * total
    acc = 0.0
    for v, w in items:
        acc += w
        if r <= acc:
            return v
    return items[-1][0]


# -----------------------------
# Model
# -----------------------------

@dataclass
class HourForecast:
    when: datetime
    temp_c: float
    feels_like_c: float
    rh_pct: int
    wind_kph: float
    wind_dir_deg: float
    cloud_pct: int
    pop_pct: int           # probability of precipitation
    precip_mm: float
    condition: str


def seasonal_baseline_oceanic(doy: int) -> float:
    """
    Rough seasonal cycle for an oceanic coastal city in mid-latitudes.
    Baseline around ~11C, annual amplitude ~4-5C.
    """
    # Peak warmth ~day 220 (Aug), coolest ~day 40 (Feb)
    # cosine: max at phase -> use shift
    amp = 4.5
    mean = 11.0
    phase = 220
    return mean + amp * math.cos(2 * math.pi * (doy - phase) / 365.25)


def diurnal_cycle(hour: int) -> float:
    """
    Small diurnal cycle: coolest near ~5am, warmest ~3pm.
    Returns a multiplier in [-1, 1].
    """
    # shift so max around 15
    return math.sin(2 * math.pi * (hour - 9) / 24.0)


def compute_hourly(
    target_date: date,
    city_name: str,
    seed: int | None = None,
) -> list[HourForecast]:
    # Deterministic by default per date (so reruns match)
    doy = day_of_year(target_date)
    base_seed = seed if seed is not None else (target_date.toordinal() * 7919)  # prime-ish
    rng = random.Random(base_seed)

    base_temp = seasonal_baseline_oceanic(doy)

    # Day "regimes": typical oceanic weather patterns
    # Higher chance of cloudy/drizzle; sometimes a rain+wind system
    regime = choose_weighted(rng, [
        ("overcast_drizzle", 0.45),
        ("broken_clouds",    0.25),
        ("showery",          0.20),
        ("sun_breaks",       0.10),
    ])

    # Sometimes a stronger frontal passage (wind + heavier rain)
    frontal = (rng.random() < 0.18)  # ~18% of days
    frontal_peak_hour = rng.randint(4, 20) if frontal else None

    # Baseline wind direction in coastal areas tends to be westerly-ish
    wdir_center = rng.uniform(230, 290)  # SW to WNW
    wind_base = rng.uniform(12, 22)      # kph

    forecasts: list[HourForecast] = []
    start = datetime(target_date.year, target_date.month, target_date.day, 0, 0)

    # Temperature day range: small (oceanic)
    diurnal_amp = rng.uniform(2.0, 4.0)  # °C

    # Humidity high-ish
    rh_base = rng.uniform(74, 88)

    for h in range(24):
        when = start + timedelta(hours=h)

        # Temperature = seasonal baseline + small diurnal + noise + regime adjustment
        di = diurnal_cycle(h)  # [-1,1]
        temp = base_temp + diurnal_amp * (0.6 * di) + rng.gauss(0, 0.4)

        if regime == "overcast_drizzle":
            temp -= 0.4
        elif regime == "sun_breaks":
            temp += 0.3 if 11 <= h <= 17 else -0.1

        # Clouds
        if regime == "sun_breaks":
            cloud = int(clamp(rng.gauss(45, 20) + (20 if h < 9 or h > 18 else -10), 5, 85))
        elif regime == "broken_clouds":
            cloud = int(clamp(rng.gauss(70, 12), 35, 95))
        elif regime == "showery":
            cloud = int(clamp(rng.gauss(80, 10), 55, 100))
        else:  # overcast_drizzle
            cloud = int(clamp(rng.gauss(92, 6), 70, 100))

        # Precip probability and amount
        pop = 0
        precip = 0.0

        # Base precip tendencies
        if regime == "sun_breaks":
            pop = int(clamp(rng.gauss(15, 10), 0, 45))
        elif regime == "broken_clouds":
            pop = int(clamp(rng.gauss(30, 12), 5, 70))
        elif regime == "showery":
            pop = int(clamp(rng.gauss(55, 15), 15, 95))
        else:  # drizzle
            pop = int(clamp(rng.gauss(65, 12), 25, 98))

        # Frontal enhancement centered on frontal_peak_hour
        if frontal and frontal_peak_hour is not None:
            # Gaussian bump around peak hour
            bump = math.exp(-0.5 * ((h - frontal_peak_hour) / 2.3) ** 2)
            pop = int(clamp(pop + 40 * bump, 0, 100))
            cloud = int(clamp(cloud + 10 * bump, 0, 100))

        # Convert PoP -> precip occurrence
        if rng.random() < (pop / 100.0):
            if regime == "overcast_drizzle":
                precip = max(0.0, rng.gauss(0.4, 0.25))  # light
            elif regime == "showery":
                precip = max(0.0, rng.gauss(1.2, 0.8))   # variable
            else:
                precip = max(0.0, rng.gauss(0.7, 0.5))

            # Frontal can intensify
            if frontal and frontal_peak_hour is not None:
                bump = math.exp(-0.5 * ((h - frontal_peak_hour) / 2.0) ** 2)
                precip *= (1.0 + 2.2 * bump)

            precip = float(clamp(precip, 0.0, 12.0))

        # Wind: base + gustiness; frontal bumps wind
        wind = wind_base + rng.gauss(0, 2.0)
        if precip > 0:
            wind += rng.uniform(0.5, 4.5)
        if frontal and frontal_peak_hour is not None:
            bump = math.exp(-0.5 * ((h - frontal_peak_hour) / 2.5) ** 2)
            wind += 18.0 * bump
        wind = float(clamp(wind, 0.0, 70.0))

        # Wind direction meanders
        wdir = (wdir_center + rng.gauss(0, 18.0)) % 360.0

        # Humidity responds to clouds/precip
        rh = rh_base + (cloud - 60) * 0.25 + (10 if precip > 0 else 0) + rng.gauss(0, 3.0)
        rh = int(clamp(rh, 45, 99))

        # Feels-like (simple wind chill effect when cool + wind)
        feels = temp
        if temp < 12 and wind > 10:
            feels -= clamp((wind - 10) * 0.03, 0, 2.5)
        if temp > 18 and rh > 75:
            feels += clamp((rh - 75) * 0.015, 0, 1.2)

        # Condition label
        if precip >= 3.0:
            cond = "Rain"
        elif precip > 0.8:
            cond = "Light rain"
        elif precip > 0.0:
            cond = "Drizzle"
        else:
            if cloud >= 90:
                cond = "Overcast"
            elif cloud >= 65:
                cond = "Mostly cloudy"
            elif cloud >= 35:
                cond = "Partly cloudy"
            else:
                cond = "Mostly clear"

        forecasts.append(HourForecast(
            when=when,
            temp_c=float(temp),
            feels_like_c=float(feels),
            rh_pct=rh,
            wind_kph=float(wind),
            wind_dir_deg=float(wdirngClamp(wdir),) if False else float(wdir),  # no-op; see note below
            cloud_pct=int(clamp(cloud, 0, 100)),
            pop_pct=int(clamp(pop, 0, 100)),
            precip_mm=float(precip),
            condition=cond
        ))

    return forecasts


def print_forecast(city: str, target_date: date, hourly: list[HourForecast]) -> None:
    print(f"{city} — Hourly forecast (oceanic climate)")
    print(f"Date: {target_date.isoformat()}\n")
    header = (
        "Time  | Temp | Feels | RH  | Wind      | Clouds | PoP | Precip | Condition"
    )
    print(header)
    print("-" * len(header))
    for hf in hourly:
        t = hf.when.strftime("%H:%M")
        wind_dir = fmt_wdir(hf.wind_dir_deg)
        print(
            f"{t} | "
            f"{hf.temp_c:4.1f}C | "
            f"{hf.feels_like_c:5.1f}C | "
            f"{hf.rh_pct:2d}% | "
            f"{hf.wind_kph:4.0f} kph {wind_dir:>3} | "
            f"{hf.cloud_pct:3d}%  | "
            f"{hf.pop_pct:3d}% | "
            f"{hf.precip_mm:5.1f}mm | "
            f"{hf.condition}"
        )


def parse_args():
    p = argparse.ArgumentParser(description="Generate an imaginary hourly oceanic-climate forecast.")
    p.add_argument("date", nargs="?", help="Date as YYYY-MM-DD (or provide year month day separately).")
    p.add_argument("month", nargs="?", type=int)
    p.add_argument("day", nargs="?", type=int)
    p.add_argument("--year", type=int, help="Year if not included in YYYY-MM-DD.")
    p.add_argument("--city", default="Kadlemoss", help="Imaginary city name.")
    p.add_argument("--seed", type=int, default=None, help="Override RNG seed for different outcomes.")
    return p.parse_args()


def main():
    args = parse_args()

    if args.date and "-" in args.date:
        d = datetime.strptime(args.date, "%Y-%m-%d").date()
    else:
        # Allow: python script.py 2026 2 16  (as args.date args.month args.day)
        if args.date is None or args.month is None or args.day is None:
            raise SystemExit("Provide a date as YYYY-MM-DD, or as: YEAR MONTH DAY")
        year = int(args.date) if args.year is None else args.year
        d = date(year, args.month, args.day)

    hourly = compute_hourly(d, args.city, seed=args.seed)
    print_forecast(args.city, d, hourly)


if __name__ == "__main__":
    main()
