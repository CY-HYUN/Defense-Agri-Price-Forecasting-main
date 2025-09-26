document.addEventListener('DOMContentLoaded', () => {
    // PowerBI 링크 설정
    const powerBiLinks = [
        "https://app.powerbi.com/reportEmbed?reportId=40f402b4-7184-46a5-9812-36f4005669a2&autoAuth=true&ctid=71c02dac-ad4f-4225-bd2d-5727e8c68a81", // GDP 차트 링크
    ];
    // iframe 요소 가져오기
    const logisticsIframe = document.getElementById('logistics-iframe');

    // 각 차트에 해당하는 링크 설정
    logisticsIframe.src = powerBiLinks[0];
    // 차트가 표시될 div 활성화
    document.getElementById('logistics-chart').classList.add('active');
});
