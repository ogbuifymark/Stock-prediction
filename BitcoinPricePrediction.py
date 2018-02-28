__author__ = 'DELL'

import sys
import pandas as pd
import time
import seaborn as sns
import matplotlib.pyplot as plt
import datetime
import numpy as np
from PIL import Image
import io


#getting market info
bitcoinMarketInfo = pd.read_html("https://coinmarketcap.com/currencies/bitcoin/historical-data/?start=20130428&end="+time.strftime("%Y%m%d"))[0]

#normalizing data
bitcoinMarketInfo = bitcoinMarketInfo.assign(Date = pd.to_datetime(bitcoinMarketInfo['Date']))
bitcoinMarketInfo.loc[bitcoinMarketInfo['Volume']=="-", 'Volume'] = 0
bitcoinMarketInfo['Volume'] = bitcoinMarketInfo['Volume'].astype('int64')
#bitcoinMarketInfo.head()

#gettung the market info for ethereum
ethMarkeInfo = pd.read_html("https://coinmarketcap.com/currencies/ethereum/historical-data/?start=20130428&end="+time.strftime("%Y%m%d"))[0]
ethMarkeInfo = ethMarkeInfo.assign(Date=pd.to_datetime(ethMarkeInfo['Date']))

if sys.version_info[0] < 3:
    import urllib2 as urllib
    bt_img = urllib.urlopen("http://logok.org/wp-content/uploads/2016/10/Bitcoin-Logo-640x480.png")
    eth_img = urllib.urlopen("https://upload.wikimedia.org/wikipedia/commons/thumb/0/05/Ethereum_logo_2014.svg/256px-Ethereum_logo_2014.svg.png")
else:
    import urllib
    bt_img = urllib.request.urlopen("http://logok.org/wp-content/uploads/2016/10/Bitcoin-Logo-640x480.png")
    eth_img = urllib.request.urlopen("https://upload.wikimedia.org/wikipedia/commons/thumb/0/05/Ethereum_logo_2014.svg/256px-Ethereum_logo_2014.svg.png")

image_file = io.BytesIO(bt_img.read())
bitcoin_im = Image.open(image_file)

image_file = io.BytesIO(eth_img.read())
eth_im = Image.open(image_file)
width_eth_im , height_eth_im  = eth_im.size
eth_im = eth_im.resize((int(eth_im.size[0]*0.6), int(eth_im.size[1]*0.6)), Image.ANTIALIAS)

bitcoinMarketInfo.columns =[bitcoinMarketInfo.columns[0]]+['bt_'+i for i in bitcoinMarketInfo.columns[1:]]
print(bitcoinMarketInfo.columns)
ethMarkeInfo.columns =[ethMarkeInfo.columns[0]]+['eth_'+i for i in ethMarkeInfo.columns[1:]]