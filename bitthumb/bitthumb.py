#! /usr/bin/env python
# XCoin API-call sample script (for Python 3.X)
#
# @author	btckorea
# @date	2017-04-11
#
#
# First, Build and install pycurl with the following commands::
# (if necessary, become root)
#
# https://pypi.python.org/pypi/pycurl/7.43.0#downloads
#
# tar xvfz pycurl-7.43.0.tar.gz
# cd pycurl-7.43.0
# python setup.py --libcurl-dll=libcurl.so install
# python setup.py --with-openssl install
# python setup.py install

from bitthumb.xcoin_api_client import *


api_key = "비밀"
api_secret = "비밀"

api = XCoinAPI(api_key, api_secret)

def getInfo():
	rgParams = {
		"currency": "xrp"
	}
	result = api.xcoinApiCall("/info/balance", rgParams)

	if result["status"] == "0000":
		return {
			"result": True,
			"xrp": result["data"]["available_xrp"],
			"krw": result["data"]["available_krw"],
			"price": result["data"]["xcoin_last"]
		}
	else:
		return {
			"result": False
		}

def bitthumb_buy(units):
	rgParams = {
		"currency": "xrp",
		"units": "%.4f" % (float(units) - 0.00005)
	}
	result = api.xcoinApiCall("/trade/market_buy", rgParams)

	if result["status"] == "0000":
		return {
			"result": True
		}
	else:
		return {
			"result": False
		}

def bitthumb_sell(xrp):
	rgParams = {
		"currency": "xrp",
		"units": "%.4f" % (float(xrp) - 0.00005)
	}
	result = api.xcoinApiCall("/trade/market_sell", rgParams)

	if result["status"] == "0000":
		return {
			"result": True
		}
	else:
		return {
			"result": False
		}
