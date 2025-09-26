from flask import Flask, render_template
import pandas as pd
import matplotlib.pyplot as plt
import io
import base64  # Base64 인코딩을 위한 모듈

# 템플릿 디렉토리를 'main'으로 설정
app = Flask(__name__, template_folder='main')

@app.route('/')
def home():
    return render_template('main_dashboard.html')

@app.route('/sub_chart_economy')
def economy():
    return render_template('sub_chart_economy.html')

@app.route('/sub_chart_logistics')
def logistics():
    return render_template('sub_chart_logistics.html')

@app.route('/sub_chart_weather')
def weather():
    return render_template('sub_chart_weather.html')

@app.route('/sub_chart_oil')
def oil():
    return render_template('sub_chart_oil.html')

@app.route('/menu_management')
def menu():
    return render_template('menu_management.html')

# EDA 페이지 (Base64로 그래프 렌더링)
@app.route('/EDA')
def EDA():
    # 데이터 생성 (예제용)
    df = pd.DataFrame({
        'Date': pd.date_range(start='2024-01-01', periods=10),
        'Value': [10, 20, 15, 25, 30, 35, 40, 30, 20, 10]
    })
    df.set_index('Date', inplace=True)

    # 그래프 생성
    plt.figure(figsize=(10, 5))
    df['Value'].plot(marker='o', title='Example Graph')
    plt.xlabel('Date')
    plt.ylabel('Value')
    plt.grid()

    # 그래프를 메모리에 저장하고 Base64 인코딩
    img = io.BytesIO()
    plt.savefig(img, format='png')
    plt.close()
    img.seek(0)
    graph_url = base64.b64encode(img.getvalue()).decode()

    # EDA.html에 그래프 전달
    return render_template('EDA.html', graph_url=graph_url)

if __name__ == '__main__':
    app.run(debug=True)