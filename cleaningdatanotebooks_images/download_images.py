# This file scraps Bing for photos of each musician. 
# Built with: https://gist.github.com/stephenhouser/c5e2b921c3770ed47eb3b75efbc94799 and chat gtp
#!/usr/bin/env python3
from bs4 import BeautifulSoup
import sys
import os
import json
import urllib.request, urllib.error, urllib.parse

def get_soup(url, header):
    return BeautifulSoup(urllib.request.urlopen(
        urllib.request.Request(url, headers=header)),
        'html.parser')

# Ensure that a valid argument is provided
if len(sys.argv) < 2:
    print("Usage: python3 download_images.py <search_query>")
    sys.exit(1)

query = sys.argv[1]
query = query.split()
query = '+'.join(query)
url = "http://www.bing.com/images/search?q=" + query + "&FORM=HDRSC2"

# Directory setup
DIR = "Pictures"
header = {'User-Agent': "Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/43.0.2357.134 Safari/537.36"}
soup = get_soup(url, header)

ActualImages = []
image_count = 0

for a in soup.find_all("a", {"class": "iusc"}):
    if image_count >= 20:
        break
    m = json.loads(a["m"])
    murl = m["murl"]
    turl = m["turl"]

    image_name = urllib.parse.urlsplit(murl).path.split("/")[-1]
    ActualImages.append((image_name, turl, murl))
    image_count += 1

print("There are total", len(ActualImages), "images")

if not os.path.exists(DIR):
    os.mkdir(DIR)

DIR = os.path.join(DIR, query.split('+')[0])
if not os.path.exists(DIR):
    os.mkdir(DIR)

for i, (image_name, turl, murl) in enumerate(ActualImages):
    try:
        raw_img = urllib.request.urlopen(turl).read()
        with open(os.path.join(DIR, image_name), 'wb') as f:
            f.write(raw_img)
    except Exception as e:
        print("Could not load:", image_name)
        print(e)
