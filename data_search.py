import os
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin
import os
import re

def sanitize_filename(filename):
    """
    清理文件名中的非法字符，替换为下划线 _。
    """
    return re.sub(r'[<>:"/\\|?*]', '_', filename)  # 替换非法字符


def download_images(url, save_folder="images"):
    """
    从指定网站爬取图片并保存到本地
    :param url: 要爬取的网页URL
    :param save_folder: 保存图片的本地文件夹
    """
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/110.0.5481.77 Safari/537.36"
    }

    # 检查保存文件夹是否存在，不存在则创建
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)

    try:
        # 发送请求获取页面内容
        response = requests.get(url, headers=headers)
        response.raise_for_status()
    except requests.exceptions.RequestException as e:
        print(f"请求失败：{e}")
        return

        # 使用 BeautifulSoup 解析 HTML
    soup = BeautifulSoup(response.text, "html.parser")

    # 查找所有的 <img> 标签
    img_tags = soup.find_all("img")

    print(f"发现了 {len(img_tags)} 张图片，开始下载...")

    for img_tag in img_tags:
        # 获取图片的 src 属性
        img_url = img_tag.get("src")

        if not img_url:
            # 如果没有 src 属性，忽略这个 <img> 标签
            continue

            # 将相对路径拼接成完整的 URL（如果 src 是绝对路径则不变）
        img_url = urljoin(url, img_url)

        try:
            # 发送请求下载图片内容
            img_data = requests.get(img_url, headers=headers).content

            # 提取图片名称
            img_name = os.path.basename(img_url)
            img_name = sanitize_filename(img_name)
            # 将图片保存到本地文件夹
            with open(os.path.join(save_folder, img_name), "wb") as f:
                f.write(img_data)

            print(f"成功下载图片：{img_name}")
        except requests.exceptions.RequestException as e:
            print(f"下载图片失败：{img_url}，原因：{e}")

    print("图片下载完成！")


# 示例调用
if __name__ == "__main__":
    # 替换为你想要爬取图片的网址
    website_url = "http://www.netbian.com/"

    # 替换为你要保存图片的本地文件夹名称
    download_folder = r"D:\ai picture"

    download_images(website_url, download_folder)