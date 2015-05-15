from bs4 import BeautifulSoup as bs
from selenium import webdriver
import re
import csv

# print soup.find(text=re.compile("Population density:")).next.strip().replace(',', '')

# soup.find(text=re.compile("Mar. 2012 cost of living index in zip code")).next.strip()

# soup.find(text=re.compile("Estimated median house/condo value in 2011")).next.strip().replace(',', '').replace('$', '')

# soup.find(text=re.compile("Median resident age:")).find_next("p").next.replace("years", "").strip()

# soup.find(text=re.compile("Average household size:")).find_next("p").next.replace("people", "").strip()

# soup.find(text=re.compile("Averages for the 2004 tax year for zip code")).find_next("b").next.next.strip().replace(',', '').replace('$', '')


def get_zips(driver):
    driver.get("http://www.city-data.com/city/New-York-New-York.html")
    soup = bs(driver.page_source)
    zips_nyc = []
    for x in soup.find(id='zip-codes').find_all("a"):
        zips_nyc.append(x.text)

    driver.get("http://www.city-data.com/city/Chicago-Illinois.html")
    soup = bs(driver.page_source)
    zips_ch = []
    for x in soup.find(id='zip-codes').find_all("a"):
        zips_ch.append(x.text)
    zips = zips_nyc[:-1] + zips_ch[:-1]
    return zips


def get_features(driver, zips):
    features = []
    features.append(['zip', 'population_density', 'cost_living', 'median_house_value',
                    'resident_age', 'household_size', 'avg_income', 'unemployment_rate'])
    for zp in zips:
        feature = []
        url = 'http://www.city-data.com/zips/' + zp + '.html'
        try:
            driver.get(url)
            soup = bs(driver.page_source)
        except:
            continue
        feature.append(zp)
        try:
            feature.append(soup.find(text=re.compile("Population density:")).next.strip().replace(',', ''))
        except:
            print zp + 'missed one'
            feature.append('')
        try:
            feature.append(soup.find(text=re.compile("Mar. 2012 cost of living index in zip code")).next.strip())
        except:
            print zp + 'missed one'
            feature.append('')
        try:
            feature.append(soup.find(text=re.compile("Estimated median house/condo value in 2011")).next.strip().replace(',', '').replace('$', ''))
        except:
            print zp + 'missed one'
            feature.append('')
        try:
            feature.append(soup.find(text=re.compile("Median resident age:")).find_next("p").next.replace("years", "").strip())
        except:
            print zp + 'missed one'
            feature.append('')
        try:
            feature.append(soup.find(text=re.compile("Average household size:")).find_next("p").next.replace("people", "").strip())
        except:
            print zp + 'missed one'
            feature.append('')
        try:
            feature.append(soup.find(text=re.compile("Averages for the 2004 tax year for zip code")).find_next("b").next.next.strip().replace(',', '').replace('$', ''))
        except:
            print zp + 'missed one'
            feature.append('')
        try:
            feature.append(soup.find(text=re.compile("Unemployed:")).next.replace('%', '').strip())
        except:
            print zp + 'missed one'
            feature.append('')
        features.append(feature)
    return features


def write_to_csv(features):
    with open("city_data.csv", "w") as f:
        writer = csv.writer(f, delimiter=',')
        for row in features:
            writer.writerow(row)
        print "zipcode data written to city_data.csv"


if __name__ == '__main__':
    uastring = 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_9_2) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/36.0.1944.0 Safari/537.36'
    dcap = webdriver.DesiredCapabilities.PHANTOMJS
    dcap["phantomjs.page.settings.userAgent"] = uastring
    exec_path = '/usr/local/bin/phantomjs'
    driver = webdriver.PhantomJS(exec_path)
    driver.set_window_size(1024, 768)
    zips = get_zips(driver)
    features = get_features(driver, zips)
    write_to_csv(features)

