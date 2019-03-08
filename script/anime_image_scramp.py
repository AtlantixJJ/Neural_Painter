"""
Download images on Safebooru and Konachan.
Proxy is enabled.

Auther: Jin Xie, Jianjin Xu.
"""
import requests
from bs4 import BeautifulSoup
import os
import traceback
import tqdm

proxies = {
        "http"  :   "http://127.0.0.1:1144",
        "https" :   "http://127.0.0.1:1144"
        }

def download(url, filename):
    if os.path.exists(filename):
        print('file exists!')
        return
    try:
        r = requests.get(url, stream=True, timeout=60, proxies=proxies)
        r.raise_for_status()
        with open(filename, 'wb') as f:
            for chunk in r.iter_content(chunk_size=1024):
                if chunk:  # filter out keep-alive new chunks
                    f.write(chunk)
                    f.flush()
        return filename
    except KeyboardInterrupt:
        if os.path.exists(filename):
            os.remove(filename)
        raise KeyboardInterrupt
    except Exception:
        traceback.print_exc()
        if os.path.exists(filename):
            os.remove(filename)

def crawler_konachan(total_page):
    img_dir = "imgs_konachan"
    if not os.path.exists(img_dir):
        os.makedirs(img_dir)

    count_success, count_fail = 0, 0
    req_failed = 0

    for i in tqdm.trange(1, total_page):
        url = 'http://konachan.net/post?page={}&tags='.format(i)
        try:
            html = requests.get(url, proxies=proxies).text
        except:
            print("Request failed")
            req_failed = req_failed + 1

        soup = BeautifulSoup(html, 'html.parser')
        #for img in soup.find_all('img', class_="preview"):
        #    target_url = 'http:' + img['src']
        #    filename = os.path.join(img_dir, target_url.split('/')[-1])
        #    download(target_url, filename)
        for thumb in soup.find_all('a'):
            try:
                url = thumb['href']
                if url.find("post") == -1 or url.find("show") == -1:
                    continue
            except:
                pass

            img_url = "http://konachan.net/" + url
            try:
                img_page = requests.get(img_url, proxies=proxies).text
            except:
                req_failed = req_failed + 1
                print("Request failed")

            img_soup = BeautifulSoup(img_page, "html.parser")

            for img in img_soup.find_all('img', class_="image"):
                target_url = 'http:' + img['src']
                filename = os.path.join(img_dir, target_url.split('/')[-1])
                download(target_url, filename)

                if os.path.exists(filename):
                    count_success += 1
                else:
                    count_fail += 1

        print('Get %d\t/\t%d\t|\tRequest failed:\t %d' % (count_success, count_success + count_fail, req_failed))


def crawler_safebooru(page_start, page_end):
    img_dir = 'imgs_safebooru'
    if not os.path.exists(img_dir):
        os.makedirs(img_dir)

    count_sucess, count_fail = (0, 0)
    for i in tqdm.trange(page_start, page_end):
        url = 'https://safebooru.org/index.php?page=post&s=list&tags=all&pid={}'.format(40 * (i-1))
        html = requests.get(url, proxies=proxies).text
        soup = BeautifulSoup(html, 'html.parser')
        for img in soup.find_all('img', class_="preview"):
            # example urls:
                # https://safebooru.org/images/137/f2d8e39ef08dced7c3f5dabdfbc45538be199899.jpg?137008
                # https://safebooru.org/thumbnails/137/thumbnail_f2d8e39ef08dced7c3f5dabdfbc45538be199899.jpg?137008
            thumbnail_url = 'http:' + img['src']
            img_url = thumbnail_url.replace('thumbnails', 'images').replace('thumbnail_', '')
            img_name = img_url.split('?')[-2].split('/')[-1]
            filename = os.path.join(img_dir, img_name)
            download(img_url, filename)
            if os.path.exists(filename):
                count_sucess += 1
            else:
                count_fail += 1
        print("sucess: {}, fail:{}, page: {}".format(count_sucess, count_fail, i))
        # print('%d / %d' % (i, total_page))

if __name__ == "__main__":
    # 9066 * 14 iamges
    crawler_konachan(9065)

    # about 2x10**6 images, 40 images/page
    # crawler_safebooru(page_start=1, page_end=1000)
    # crawler_safebooru(page_start=1000, page_end=2000)
