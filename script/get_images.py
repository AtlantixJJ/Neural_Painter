import urllib
import urllib2
import os

DOMAIN = 'http://166.111.17.31:8080'
TAG = 'generated/image/0'
RUN = 'goodmodel_wgan_getchu3'
PATH = '/data/plugin/images/individualImage'
DIR = RUN
MAX = 220
RANGE = [21, 60, 102, 202]

if __name__ == '__main__':
    if not os.path.exists(DIR):
        os.mkdir(DIR)

    for i in RANGE:
        index = i * 100 / MAX
        data = urllib.urlencode({
            'index': index,
            'tag': TAG,
            'run': RUN,
        })

        url = DOMAIN + PATH + '?' + data
        print("index: %d url: %s" % (i, url))
        req = urllib.urlopen(url)
        img = req.read()

        f = open(os.path.join(DIR, '%s_%d.png' % (RUN, index)), 'w')
        f.write(img)
        f.close()
