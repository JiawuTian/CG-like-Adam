import requests


def send_notice(token:str, title:str="主程序结束", content:str="训练代码主程序结束"):
    url = f"http://www.pushplus.plus/send?token={token}&title={title}&content={content}&template=html"
    response = requests.request("GET", url)
    print(response.text)



if __name__ == "__main__":
    send_notice(token="",)


