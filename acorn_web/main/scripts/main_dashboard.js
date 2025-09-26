// 그래프 렌더링
document.addEventListener('DOMContentLoaded', () => {
    const lstm_sarimax_Iframe = document.getElementById('lstm-sarimax-iframe');

    // 각 제품에 대한 Power BI 링크
    const powerBiLinks = [
        "https://app.powerbi.com/reportEmbed?reportId=2708d873-dbd3-49b5-bbf0-d0578d2e7600&autoAuth=true&ctid=71c02dac-ad4f-4225-bd2d-5727e8c68a81", // 페이지1
    ];

    lstm_sarimax_Iframe.src = powerBiLinks[0];
    document.getElementById('lstm-sarimax-chart').classList.add('active');
});
