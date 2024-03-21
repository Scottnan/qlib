# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import abc
import sys
import copy
import time
import datetime
import importlib
from abc import ABC
import multiprocessing
from pathlib import Path
from typing import Iterable

import fire
import requests
import numpy as np
import pandas as pd
from loguru import logger
from dateutil.tz import tzlocal
from WindPy import w

import qlib
from qlib.data import D
from qlib.tests.data import GetData
from qlib.utils import code_to_fname, fname_to_code, exists_qlib_data
from qlib.constant import REG_CN as REGION_CN

CUR_DIR = Path(__file__).resolve().parent
sys.path.append(str(CUR_DIR.parent.parent))
w.start()

from dump_bin import DumpDataUpdate
from data_collector.base import BaseCollector, BaseNormalize, BaseRun, Normalize
from data_collector.utils import (
    deco_retry,
    get_calendar_list,
    get_hs_stock_symbols,
    get_us_stock_symbols,
    get_in_stock_symbols,
    get_br_stock_symbols,
    generate_minutes_calendar_from_daily,
    calc_adjusted_price,
)

INDEX_BENCH_URL = "http://push2his.eastmoney.com/api/qt/stock/kline/get?secid=1.{index_code}&fields1=f1%2Cf2%2Cf3%2Cf4%2Cf5&fields2=f51%2Cf52%2Cf53%2Cf54%2Cf55%2Cf56%2Cf57%2Cf58&klt=101&fqt=0&beg={begin}&end={end}"


class WindCollector(BaseCollector):
    retry = 5  # Configuration attribute.  How many times will it try to re-request the data if the network fails.

    def __init__(
        self,
        save_dir: [str, Path],
        start=None,
        end=None,
        interval="1d",
        max_workers=4,
        max_collector_count=2,
        delay=0,
        check_data_length: int = None,
        limit_nums: int = None,
    ):
        """

        Parameters
        ----------
        save_dir: str
            stock save dir
        max_workers: int
            workers, default 4
        max_collector_count: int
            default 2
        delay: float
            time.sleep(delay), default 0
        interval: str
            freq, value from [1min, 1d], default 1min
        start: str
            start datetime, default None
        end: str
            end datetime, default None
        check_data_length: int
            check data length, by default None
        limit_nums: int
            using for debug, by default None
        """
        super(WindCollector, self).__init__(
            save_dir=save_dir,
            start=start,
            end=end,
            interval=interval,
            max_workers=max_workers,
            max_collector_count=max_collector_count,
            delay=delay,
            check_data_length=check_data_length,
            limit_nums=limit_nums,
        )

        self.init_datetime()

    def init_datetime(self):
        if self.interval == self.INTERVAL_1min:
            self.start_datetime = max(self.start_datetime, self.DEFAULT_START_DATETIME_1MIN)
        elif self.interval == self.INTERVAL_1d:
            pass
        else:
            raise ValueError(f"interval error: {self.interval}")

        self.start_datetime = self.convert_datetime(self.start_datetime, self._timezone)
        self.end_datetime = self.convert_datetime(self.end_datetime, self._timezone)

    @staticmethod
    def convert_datetime(dt: [pd.Timestamp, datetime.date, str], timezone):
        try:
            dt = pd.Timestamp(dt, tz=timezone).timestamp()
            dt = pd.Timestamp(dt, tz=tzlocal(), unit="s")
        except ValueError as e:
            pass
        return dt

    @property
    @abc.abstractmethod
    def _timezone(self):
        raise NotImplementedError("rewrite get_timezone")

    @abc.abstractmethod
    @staticmethod
    def get_data_from_remote(symbol, interval, start, end, show_1min_logging: bool = False):
        pass
        # error_msg = f"{symbol}-{interval}-{start}-{end}"
        #
        # def _show_logging_func():
        #     if interval == WindCollector.INTERVAL_1min and show_1min_logging:
        #         logger.warning(f"{error_msg}:{_resp}")
        #
        # interval = "1m" if interval in ["1m", "1min"] else interval
        # try:
        #     _resp = Ticker(symbol, asynchronous=False).history(interval=interval, start=start, end=end)
        #     if isinstance(_resp, pd.DataFrame):
        #         return _resp.reset_index()
        #     elif isinstance(_resp, dict):
        #         _temp_data = _resp.get(symbol, {})
        #         if isinstance(_temp_data, str) or (
        #             isinstance(_resp, dict) and _temp_data.get("indicators", {}).get("quote", None) is None
        #         ):
        #             _show_logging_func()
        #     else:
        #         _show_logging_func()
        # except Exception as e:
        #     logger.warning(
        #         f"get data error: {symbol}--{start}--{end}"
        #         + "Your data request fails. This may be caused by your firewall (e.g. GFW). Please switch your network if you want to access Yahoo! data"
        #     )

    def get_data(
        self, symbol: str, interval: str, start_datetime: pd.Timestamp, end_datetime: pd.Timestamp
    ) -> pd.DataFrame:
        @deco_retry(retry_sleep=self.delay, retry=self.retry)
        def _get_simple(start_, end_):
            self.sleep()
            _remote_interval = "1m" if interval == self.INTERVAL_1min else interval
            resp = self.get_data_from_remote(
                symbol,
                interval=_remote_interval,
                start=start_,
                end=end_,
            )
            if resp is None or resp.empty:
                raise ValueError(
                    f"get data error: {symbol}--{start_}--{end_}" + "The stock may be delisted, please check"
                )
            return resp

        _result = None
        if interval == self.INTERVAL_1d:
            try:
                _result = _get_simple(start_datetime, end_datetime)
            except ValueError as e:
                pass
        elif interval == self.INTERVAL_1min:
            _res = []
            _start = self.start_datetime
            while _start < self.end_datetime:
                _tmp_end = min(_start + pd.Timedelta(days=7), self.end_datetime)
                try:
                    _resp = _get_simple(_start, _tmp_end)
                    _res.append(_resp)
                except ValueError as e:
                    pass
                _start = _tmp_end
            if _res:
                _result = pd.concat(_res, sort=False).sort_values(["symbol", "date"])
        else:
            raise ValueError(f"cannot support {self.interval}")
        return pd.DataFrame() if _result is None else _result

    def collector_data(self):
        """collector data"""
        super(WindCollector, self).collector_data()
        self.download_index_data()

    @abc.abstractmethod
    def download_index_data(self):
        """download index data"""
        raise NotImplementedError("rewrite download_index_data")


class WindCollectorCN1d(WindCollector):
    """
    This function is used to download A-share data
    """
    @staticmethod
    def get_data_from_remote(
        symbol, interval, start, end, show_1min_logging: bool = False
    ) -> pd.DataFrame:
        error_msg = f"{symbol}-{interval}-{start}-{end}"
        try:
            _data = w.wsd(symbol, "open,close,high,low,volume", start, end, "PriceAdj=B", usedf=True)
            if isinstance(_data, pd.DataFrame):
                return _data.reset_index()
        except Exception as e:
            logger.warning(
                f"get data error: {symbol}--{start}--{end}"
                + "Your data request fails."
            )

    def download_index_data(self):
        _format = "%Y-%m-%d"
        _begin = self.start_datetime.strftime(_format)
        _end = self.end_datetime.strftime(_format)
        for _index_name, _index_code in {"csi300": "000300.SH", "csi500": "000905.SH", "WindEHF": "885001.WI"}.items():
            # Wind Equity Hybrid Fund: 885001.WI
            logger.info(f"get bench data: {_index_name}({_index_code})......")
            try:
                df = w.wsd(_index_code, "open,close,high,low,volume", _begin, _end, "")
            except Exception as e:
                logger.warning(f"get {_index_name} error: {e}")
                continue
            df.columns = ["date", "open", "close", "high", "low", "volume"]
            df["date"] = pd.to_datetime(df["date"])
            df = df.astype(float, errors="ignore")
            df["adjclose"] = df["close"]
            _index_code_suffix = _index_code.split(".")
            df["symbol"] = f"{_index_code_suffix[1]}{_index_code_suffix[0]}"
            _path = self.save_dir.joinpath(f"{_index_code_suffix[1]}{_index_code_suffix[0]}.csv")
            if _path.exists():
                _old_df = pd.read_csv(_path)
                df = pd.concat([_old_df, df], sort=False)
            df.to_csv(_path, index=False)
            time.sleep(5)

    @property
    def _timezone(self):
        return "Asia/Shanghai"


class WindNormalize(BaseNormalize):
    COLUMNS = ["open", "close", "high", "low", "volume"]
    DAILY_FORMAT = "%Y-%m-%d"

    @staticmethod
    def normalize_wind(
        df: pd.DataFrame,
        calendar_list: list = None,
        date_field_name: str = "date",
        symbol_field_name: str = "symbol",
        last_close: float = None,
    ):
        if df.empty:
            return df
        symbol = df.loc[df[symbol_field_name].first_valid_index(), symbol_field_name]
        columns = copy.deepcopy(WindNormalize.COLUMNS)
        df = df.copy()
        df.set_index(date_field_name, inplace=True)
        df.index = pd.to_datetime(df.index)
        df.index = df.index.tz_localize(None)
        df = df[~df.index.duplicated(keep="first")]
        if calendar_list is not None:
            df = df.reindex(
                pd.DataFrame(index=calendar_list)
                .loc[
                    pd.Timestamp(df.index.min()).date() : pd.Timestamp(df.index.max()).date()
                    + pd.Timedelta(hours=23, minutes=59)
                ]
                .index
            )
        df.sort_index(inplace=True)
        df.loc[(df["volume"] <= 0) | np.isnan(df["volume"]), list(set(df.columns) - {symbol_field_name})] = np.nan
        df.loc[(df["volume"] <= 0) | np.isnan(df["volume"]), columns] = np.nan

        df[symbol_field_name] = symbol
        df.index.names = [date_field_name]
        return df.reset_index()

    def normalize(self, df: pd.DataFrame) -> pd.DataFrame:
        # normalize
        df = self.normalize_yahoo(df, self._calendar_list, self._date_field_name, self._symbol_field_name)
        # adjusted price
        df = self.adjusted_price(df)
        return df

    @abc.abstractmethod
    def adjusted_price(self, df: pd.DataFrame) -> pd.DataFrame:
        """adjusted price"""
        raise NotImplementedError("rewrite adjusted_price")


class WindNormalize1d(WindNormalize, ABC):
    DAILY_FORMAT = "%Y-%m-%d"

    def adjusted_price(self, df: pd.DataFrame) -> pd.DataFrame:
        if df.empty:
            return df
        df = df.copy()
        df.set_index(self._date_field_name, inplace=True)
        if "adjclose" in df:
            df["factor"] = df["adjclose"] / df["close"]
            df["factor"] = df["factor"].fillna(method="ffill")
        else:
            df["factor"] = 1
        for _col in self.COLUMNS:
            if _col not in df.columns:
                continue
            if _col == "volume":
                df[_col] = df[_col] / df["factor"]
            else:
                df[_col] = df[_col] * df["factor"]
        df.index.names = [self._date_field_name]
        return df.reset_index()

    def normalize(self, df: pd.DataFrame) -> pd.DataFrame:
        df = super(WindNormalize1d, self).normalize(df)
        df = self._manual_adj_data(df)
        return df

    def _get_first_close(self, df: pd.DataFrame) -> float:
        """get first close value

        Notes
        -----
            For incremental updates(append) to Yahoo 1D data, user need to use a close that is not 0 on the first trading day of the existing data
        """
        df = df.loc[df["close"].first_valid_index() :]
        _close = df["close"].iloc[0]
        return _close

    def _manual_adj_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """manual adjust data: All fields (except change) are standardized according to the close of the first day"""
        if df.empty:
            return df
        df = df.copy()
        df.sort_values(self._date_field_name, inplace=True)
        df = df.set_index(self._date_field_name)
        _close = self._get_first_close(df)
        for _col in df.columns:
            # NOTE: retain original adjclose, required for incremental updates
            if _col in [self._symbol_field_name, "adjclose", "change"]:
                continue
            if _col == "volume":
                df[_col] = df[_col] * _close
            else:
                df[_col] = df[_col] / _close
        return df.reset_index()


class WindNormalize1dExtend(WindNormalize1d):
    def __init__(
        self, old_qlib_data_dir: [str, Path], date_field_name: str = "date", symbol_field_name: str = "symbol", **kwargs
    ):
        """

        Parameters
        ----------
        old_qlib_data_dir: str, Path
            the qlib data to be updated for Wind
        date_field_name: str
            date field name, default is date
        symbol_field_name: str
            symbol field name, default is symbol
        """
        super(WindNormalize1dExtend, self).__init__(date_field_name, symbol_field_name)
        self.column_list = ["open", "high", "low", "close", "volume", "factor"]
        self.old_qlib_data = self._get_old_data(old_qlib_data_dir)

    def _get_old_data(self, qlib_data_dir: [str, Path]):
        qlib_data_dir = str(Path(qlib_data_dir).expanduser().resolve())
        qlib.init(provider_uri=qlib_data_dir, expression_cache=None, dataset_cache=None)
        df = D.features(D.instruments("all"), ["$" + col for col in self.column_list])
        df.columns = self.column_list
        return df

    def normalize(self, df: pd.DataFrame) -> pd.DataFrame:
        df = super(WindNormalize1dExtend, self).normalize(df)
        df.set_index(self._date_field_name, inplace=True)
        symbol_name = df[self._symbol_field_name].iloc[0]
        old_symbol_list = self.old_qlib_data.index.get_level_values("instrument").unique().to_list()
        if str(symbol_name).upper() not in old_symbol_list:
            return df.reset_index()
        old_df = self.old_qlib_data.loc[str(symbol_name).upper()]
        latest_date = old_df.index[-1]
        df = df.loc[latest_date:]
        new_latest_data = df.iloc[0]
        old_latest_data = old_df.loc[latest_date]
        for col in self.column_list[:-1]:
            if col == "volume":
                df[col] = df[col] / (new_latest_data[col] / old_latest_data[col])
            else:
                df[col] = df[col] * (old_latest_data[col] / new_latest_data[col])
        return df.drop(df.index[0]).reset_index()


class Run(BaseRun):
    def __init__(self, source_dir=None, normalize_dir=None, max_workers=1, interval="1d", region=REGION_CN):
        """

        Parameters
        ----------
        source_dir: str
            The directory where the raw data collected from the Internet is saved, default "Path(__file__).parent/source"
        normalize_dir: str
            Directory for normalize data, default "Path(__file__).parent/normalize"
        max_workers: int
            Concurrent number, default is 1; when collecting data, it is recommended that max_workers be set to 1
        interval: str
            freq, value from [1min, 1d], default 1d
        region: str
            region, value from ["CN", "US", "BR"], default "CN"
        """
        super().__init__(source_dir, normalize_dir, max_workers, interval)
        self.region = region

    @property
    def collector_class_name(self):
        return f"WindCollector{self.region.upper()}{self.interval}"

    @property
    def normalize_class_name(self):
        return f"WindNormalize{self.region.upper()}{self.interval}"

    @property
    def default_base_dir(self) -> [Path, str]:
        return CUR_DIR

    def download_data(
        self,
        max_collector_count=2,
        delay=0.5,
        start=None,
        end=None,
        check_data_length=None,
        limit_nums=None,
    ):
        """download data from Internet

        Parameters
        ----------
        max_collector_count: int
            default 2
        delay: float
            time.sleep(delay), default 0.5
        start: str
            start datetime, default "2000-01-01"; closed interval(including start)
        end: str
            end datetime, default ``pd.Timestamp(datetime.datetime.now() + pd.Timedelta(days=1))``; open interval(excluding end)
        check_data_length: int
            check data length, if not None and greater than 0, each symbol will be considered complete if its data length is greater than or equal to this value, otherwise it will be fetched again, the maximum number of fetches being (max_collector_count). By default None.
        limit_nums: int
            using for debug, by default None

        Notes
        -----
            check_data_length, example:
                daily, one year: 252 // 4
                us 1min, a week: 6.5 * 60 * 5
                cn 1min, a week: 4 * 60 * 5

        Examples
        ---------
            # get daily data
            $ python collector.py download_data --source_dir ~/.qlib/stock_data/source --region CN --start 2020-11-01 --end 2020-11-10 --delay 0.1 --interval 1d
            # get 1m data
            $ python collector.py download_data --source_dir ~/.qlib/stock_data/source --region CN --start 2020-11-01 --end 2020-11-10 --delay 0.1 --interval 1m
        """
        super(Run, self).download_data(max_collector_count, delay, start, end, check_data_length, limit_nums)

    def normalize_data(
        self,
        date_field_name: str = "date",
        symbol_field_name: str = "symbol",
        end_date: str = None,
        qlib_data_1d_dir: str = None,
    ):
        """normalize data

        Parameters
        ----------
        date_field_name: str
            date field name, default date
        symbol_field_name: str
            symbol field name, default symbol
        end_date: str
            if not None, normalize the last date saved (including end_date); if None, it will ignore this parameter; by default None
        qlib_data_1d_dir: str
            if interval==1min, qlib_data_1d_dir cannot be None, normalize 1min needs to use 1d data;

                qlib_data_1d can be obtained like this:
                    $ python scripts/get_data.py qlib_data --target_dir <qlib_data_1d_dir> --interval 1d
                    $ python scripts/data_collector/yahoo/collector.py update_data_to_bin --qlib_data_1d_dir <qlib_data_1d_dir> --trading_date 2021-06-01
                or:
                    download 1d data, reference: https://github.com/microsoft/qlib/tree/main/scripts/data_collector/yahoo#1d-from-yahoo

        Examples
        ---------
            $ python collector.py normalize_data --source_dir ~/.qlib/stock_data/source --normalize_dir ~/.qlib/stock_data/normalize --region cn --interval 1d
            $ python collector.py normalize_data --qlib_data_1d_dir ~/.qlib/qlib_data/cn_data --source_dir ~/.qlib/stock_data/source_cn_1min --normalize_dir ~/.qlib/stock_data/normalize_cn_1min --region CN --interval 1min
        """
        if self.interval.lower() == "1min":
            if qlib_data_1d_dir is None or not Path(qlib_data_1d_dir).expanduser().exists():
                raise ValueError(
                    "If normalize 1min, the qlib_data_1d_dir parameter must be set: --qlib_data_1d_dir <user qlib 1d data >, Reference: https://github.com/microsoft/qlib/tree/main/scripts/data_collector/yahoo#automatic-update-of-daily-frequency-datafrom-yahoo-finance"
                )
        super(Run, self).normalize_data(
            date_field_name, symbol_field_name, end_date=end_date, qlib_data_1d_dir=qlib_data_1d_dir
        )

    def normalize_data_1d_extend(
        self, old_qlib_data_dir, date_field_name: str = "date", symbol_field_name: str = "symbol"
    ):
        """normalize data extend; extending Wind data

        Notes
        -----
            Steps to extend Wind qlib data:

                1. download qlib data: https://github.com/microsoft/qlib/tree/main/scripts#download-cn-data; save to <dir1>

                2. collector source data: https://github.com/microsoft/qlib/tree/main/scripts/data_collector/yahoo#collector-data; save to <dir2>

                3. normalize new source data(from step 2): python scripts/data_collector/yahoo/collector.py normalize_data_1d_extend --old_qlib_dir <dir1> --source_dir <dir2> --normalize_dir <dir3> --region CN --interval 1d

                4. dump data: python scripts/dump_bin.py dump_update --csv_path <dir3> --qlib_dir <dir1> --freq day --date_field_name date --symbol_field_name symbol --exclude_fields symbol,date

                5. update instrument(eg. csi300): python python scripts/data_collector/cn_index/collector.py --index_name CSI300 --qlib_dir <dir1> --method parse_instruments

        Parameters
        ----------
        old_qlib_data_dir: str
            the qlib data to be updated for yahoo, usually from: https://github.com/microsoft/qlib/tree/main/scripts#download-cn-data
        date_field_name: str
            date field name, default date
        symbol_field_name: str
            symbol field name, default symbol

        Examples
        ---------
            $ python collector.py normalize_data_1d_extend --old_qlib_dir ~/.qlib/qlib_data/cn_data --source_dir ~/.qlib/stock_data/source --normalize_dir ~/.qlib/stock_data/normalize --region CN --interval 1d
        """
        _class = getattr(self._cur_module, f"{self.normalize_class_name}Extend")
        yc = Normalize(
            source_dir=self.source_dir,
            target_dir=self.normalize_dir,
            normalize_class=_class,
            max_workers=self.max_workers,
            date_field_name=date_field_name,
            symbol_field_name=symbol_field_name,
            old_qlib_data_dir=old_qlib_data_dir,
        )
        yc.normalize()

    def download_today_data(
        self,
        max_collector_count=2,
        delay=0.5,
        check_data_length=None,
        limit_nums=None,
    ):
        """download today data from Internet

        Parameters
        ----------
        max_collector_count: int
            default 2
        delay: float
            time.sleep(delay), default 0.5
        check_data_length: int
            check data length, if not None and greater than 0, each symbol will be considered complete if its data length is greater than or equal to this value, otherwise it will be fetched again, the maximum number of fetches being (max_collector_count). By default None.
        limit_nums: int
            using for debug, by default None

        Notes
        -----
            Download today's data:
                start_time = datetime.datetime.now().date(); closed interval(including start)
                end_time = pd.Timestamp(start_time + pd.Timedelta(days=1)).date(); open interval(excluding end)

            check_data_length, example:
                daily, one year: 252 // 4
                us 1min, a week: 6.5 * 60 * 5
                cn 1min, a week: 4 * 60 * 5

        Examples
        ---------
            # get daily data
            $ python collector.py download_today_data --source_dir ~/.qlib/stock_data/source --region CN --delay 0.1 --interval 1d
            # get 1m data
            $ python collector.py download_today_data --source_dir ~/.qlib/stock_data/source --region CN --delay 0.1 --interval 1m
        """
        start = datetime.datetime.now().date()
        end = pd.Timestamp(start + pd.Timedelta(days=1)).date()
        self.download_data(
            max_collector_count,
            delay,
            start.strftime("%Y-%m-%d"),
            end.strftime("%Y-%m-%d"),
            check_data_length,
            limit_nums,
        )

    def update_data_to_bin(
        self,
        qlib_data_1d_dir: str,
        end_date: str = None,
        check_data_length: int = None,
        delay: float = 1,
        exists_skip: bool = False,
    ):
        """update Wind data to bin

        Parameters
        ----------
        qlib_data_1d_dir: str
            the qlib data to be updated for Wind

        end_date: str
            end datetime, default ``pd.Timestamp(trading_date + pd.Timedelta(days=1))``; open interval(excluding end)
        check_data_length: int
            check data length, if not None and greater than 0, each symbol will be considered complete if its data length is greater than or equal to this value, otherwise it will be fetched again, the maximum number of fetches being (max_collector_count). By default None.
        delay: float
            time.sleep(delay), default 1
        exists_skip: bool
            exists skip, by default False
        Notes
        -----
            If the data in qlib_data_dir is incomplete, np.nan will be populated to trading_date for the previous trading day

        Examples
        -------
            $ python collector.py update_data_to_bin --qlib_data_1d_dir <user data dir> --trading_date <start date> --end_date <end date>
        """

        if self.interval.lower() != "1d":
            logger.warning(f"currently supports 1d data updates: --interval 1d")

        # download qlib 1d data
        qlib_data_1d_dir = str(Path(qlib_data_1d_dir).expanduser().resolve())
        if not exists_qlib_data(qlib_data_1d_dir):
            GetData().qlib_data(
                target_dir=qlib_data_1d_dir, interval=self.interval, region=self.region, exists_skip=exists_skip
            )

        # start/end date
        calendar_df = pd.read_csv(Path(qlib_data_1d_dir).joinpath("calendars/day.txt"))
        trading_date = (pd.Timestamp(calendar_df.iloc[-1, 0]) - pd.Timedelta(days=1)).strftime("%Y-%m-%d")

        if end_date is None:
            end_date = (pd.Timestamp(trading_date) + pd.Timedelta(days=1)).strftime("%Y-%m-%d")

        # download data from wind
        # NOTE: when downloading data from YahooFinance, max_workers is recommended to be 1
        self.download_data(delay=delay, start=trading_date, end=end_date, check_data_length=check_data_length)
        # NOTE: a larger max_workers setting here would be faster
        self.max_workers = (
            max(multiprocessing.cpu_count() - 2, 1)
            if self.max_workers is None or self.max_workers <= 1
            else self.max_workers
        )
        # normalize data
        self.normalize_data_1d_extend(qlib_data_1d_dir)

        # dump bin
        _dump = DumpDataUpdate(
            csv_path=self.normalize_dir,
            qlib_dir=qlib_data_1d_dir,
            exclude_fields="symbol,date",
            max_workers=self.max_workers,
        )
        _dump.dump()

        # parse index
        _region = self.region.lower()
        if _region not in ["cn", "us"]:
            logger.warning(f"Unsupported region: region={_region}, component downloads will be ignored")
            return
        index_list = ["CSI100", "CSI300"] if _region == "cn" else ["SP500", "NASDAQ100", "DJIA", "SP400"]
        get_instruments = getattr(
            importlib.import_module(f"data_collector.{_region}_index.collector"), "get_instruments"
        )
        for _index in index_list:
            get_instruments(str(qlib_data_1d_dir), _index, market_index=f"{_region}_index")


if __name__ == "__main__":
    fire.Fire(Run)
