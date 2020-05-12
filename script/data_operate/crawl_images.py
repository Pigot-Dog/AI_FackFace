import urllib
from bs4 import BeautifulSoup

def crawl_imgs():
    html = urllib.request.urlopen('https://cn.bing.com/images/search?q=%e6%9d%a8%e8%b6%85%e8%b6%8a&FORM=HDRSC2').read()
    soup = BeautifulSoup(html, 'html.parser', from_encoding='utf-8')

    images = soup.findAll('img')
    print(images)
    imageName = 0
    for image in images:
        if image.has_attr('src') and image['src'].startswith('https://'):
            image_url = image['src']
            # link = image.link('src')
            print(image_url)
            # link = 'https://'+ link
            fileSavePath = '/home/maxingpei/AI-FakeFace/train_face/chaoyue/' + str(imageName) + '.jpg'
            imageName = imageName +1
            urllib.request.urlretrieve(image_url, fileSavePath)


if __name__ == '__main__':
    crawl_imgs()