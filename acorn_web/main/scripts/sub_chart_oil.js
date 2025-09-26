document.addEventListener('DOMContentLoaded', () => {
    // PowerBI 링크 설정
    const powerBiLinks = {
        economy: {
            changeChart: "https://app.powerbi.com/view?r=eyJrIjoiXXXXX1&embed=true", // GDP 차트 링크
            monthChart: "https://app.powerbi.com/view?r=eyJrIjoiXXXXX2&embed=true" // 금리 차트 링크
        }
    };

    // iframe 요소 가져오기
    const changeIframe = document.getElementById('oil-change-iframe');
    const monthIframe = document.getElementById('oil-month-iframe');

    // 각 차트에 해당하는 링크 설정
    changeIframe.src = powerBiLinks.economy.changeChart;
    monthIframe.src = powerBiLinks.economy.monthChart;

    // 차트가 표시될 div 활성화
    document.getElementById('oil-change-chart').classList.add('active');
    document.getElementById('oil-month-chart').classList.add('active');
});
