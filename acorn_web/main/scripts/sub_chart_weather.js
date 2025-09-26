// 그래프 렌더링
document.addEventListener('DOMContentLoaded', () => {
    const weather_Iframe = document.getElementById('weather-iframe');

    // 각 제품에 대한 Power BI 링크
    const powerBiLinks = [
        "https://app.powerbi.com/reportEmbed?reportId=dbac05d0-5e4d-4ea2-8d02-ceae21c79a0d&autoAuth=true&ctid=71c02dac-ad4f-4225-bd2d-5727e8c68a81"
    ];

    weather_Iframe.src = powerBiLinks[0];
});