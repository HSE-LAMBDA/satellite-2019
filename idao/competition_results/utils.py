import numpy as np
import math
import orbital
import tletools
from sgp4.earth_gravity import wgs72
from sgp4.io import twoline2rv
from sgp4.ext import rv2coe
import datetime
import pandas as pd

MU = 3.986004419e+14


def checksum(line):
    """Performs the checksum for the current TLE."""
    check = 0
    for char in line:
        if char.isdigit():
            check += int(char)
        if char == "-":
            check += 1
    return check % 10


def bstar2tle(bstar):
    if float(bstar) >= 1:
        return "00000-0"
    if not isinstance(bstar, str):
        bstar = str(bstar)
    if len(bstar) > 2:
        bstar = bstar[2:]
        for i in range(1, 9):
            if len(bstar) == 0 or bstar[0] == "0":
                bstar = bstar[1:]
            else:
                if len(bstar) >= 5:
                    bstar = bstar[:5]
                else:
                    bstar += "0" * (5 - len(bstar))

                return bstar + "-" + str(i - 1)
    return "00000-0"


# TODO: style

def _to_tle_format(p):
    if p >= 100:
        return f"{p:.4f}"
    elif p >= 10:
        return f"0{p:.4f}"
    elif p >= 0:
        return f"00{p:.4f}"


def _n_tle_format(n):
    if n < 10:
        return f"0{n:.8f}"
    return f"{n:.8f}"


def _day_tle_format(d):
    if d >= 100:
        return f"{d:.8f}"
    elif d >= 10:
        return f"0{d:.8f}"
    elif d >= 0:
        return f"00{d:.8f}"


def _t2f(x, is_sec=False):
    if is_sec:
        x = f"{x:.3f}"
    if float(x) < 10:
        x = "0" + str(x)
    return str(x)


def tle2tle_lines(tle):
    line1 = (
        f'1 {tle.norad}{tle.classification} {tle.int_desig}  {str(tle.epoch_year)[1:]}{_day_tle_format(tle.epoch_day)}  .00000000  00000-0  {bstar2tle(tle.bstar)} 0  {tle.set_num}')
    line1 += str(checksum(line1))
    line2 = (
        f"2 {tle.norad} {_to_tle_format(tle.inc)} {_to_tle_format(tle.raan)} {f'{tle.ecc:.7f}'[2:]} {_to_tle_format(tle.argp)} {_to_tle_format(tle.M)} {_n_tle_format(tle.n)}{tle.rev_num}")
    line2 += str(checksum(line2))
    return line1, line2


def _t_orbit(SMA):
    return 2 * np.pi * ((SMA * 1000) ** 3 / MU) ** 0.5


def _mean_motion(SMA):
    return 24 * 60 * 60 / _t_orbit(SMA)


def epoch2datetime(epoch):
    return datetime.datetime.strptime(epoch, '%Y-%m-%dT%H:%M:%S.%f')


def epoch2day_year(date):
    year = date.year
    temp = datetime.datetime(year, 1, 1)
    diff = date - temp
    day = diff.total_seconds() / (60 * 60 * 24)
    return year, day + 1


def params2tle(ref_epoch, inc, raan, ecc, arg_p, mean_an, sma):
    ref_date = epoch2datetime(ref_epoch)
    epoch_year, epoch_day = epoch2day_year(ref_date)
    return tletools.TLE(
        name="sat",
        norad='99999',
        classification='U',
        int_desig='99999A',
        epoch_year=epoch_year,
        epoch_day=epoch_day,
        dn_o2=0.00000000,
        ddn_o6=0.0,
        bstar=0,
        set_num=999,
        inc=inc,
        raan=raan,
        ecc=ecc,
        argp=arg_p,
        M=mean_an,
        n=_mean_motion(sma),
        rev_num=99999
    )


def sgp4_ephemeris(coords, prediction_dates_list):
    r = coords[['x', 'y', 'z']].values
    v = coords[['Vx', 'Vy', 'Vz']].values
    ref_date = coords["epoch"]

    p, sma, ecc, inc, raan, arg_p, nu, mean_an, arglat, truelon, lonper = rv2coe(r, v, wgs72.mu)
    inc = math.degrees(inc)
    raan = math.degrees(raan)
    arg_p = math.degrees(arg_p)
    mean_an = math.degrees(mean_an)

    tle = params2tle(ref_epoch=ref_date, inc=inc, raan=raan, ecc=ecc, arg_p=arg_p, mean_an=mean_an, sma=sma)
    lines = tle2tle_lines(tle)
    sat = twoline2rv(lines[0], lines[1], wgs72)

    ephemeris = []
    for date in prediction_dates_list:
        date_ = epoch2datetime(date)
        y_ = date_.year
        m_ = date_.month
        d_ = date_.day
        h_ = date_.hour
        min_ = date_.minute
        sec_ = date_.second + date_.microsecond * 1e-6
        r, v = sat.propagate(
            y_, m_, d_, h_, min_, sec_)
        ephemeris.append([date] + list(r) + list(v))

    return pd.DataFrame(data=ephemeris, columns=['epoch'] + [c + "_sim_upd" for c in ['x', 'y', 'z', 'Vx', 'Vy', 'Vz']])
