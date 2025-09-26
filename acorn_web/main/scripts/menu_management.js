document.addEventListener('DOMContentLoaded', () => {
    // PowerBI 링크 설정
    const powerBiLinks = {
        btn1: "https://app.powerbi.com/reportEmbed?reportId=35152bd6-146d-46d0-b6c8-b674dae4fe28&autoAuth=true&ctid=71c02dac-ad4f-4225-bd2d-5727e8c68a81", // GDP 차트 링크
        btn2: "https://app.powerbi.com/reportEmbed?reportId=65f03866-8817-4b0d-9dbe-7d8aacd50436&autoAuth=true&ctid=71c02dac-ad4f-4225-bd2d-5727e8c68a81", // GDP 차트 링크
        btn3: "https://app.powerbi.com/reportEmbed?reportId=9e742bec-5aaf-4b30-9b1d-5ca24a8b7cda&autoAuth=true&ctid=71c02dac-ad4f-4225-bd2d-5727e8c68a81", // GDP 차트 링크
        btn4: "https://app.powerbi.com/reportEmbed?reportId=3e0d792b-7ef7-4a70-abed-b604d69314ab&autoAuth=true&ctid=71c02dac-ad4f-4225-bd2d-5727e8c68a81", // GDP 차트 링크
        btn5: "https://app.powerbi.com/reportEmbed?reportId=7e177cfb-f278-4a7c-96d3-19b5a5c79b35&autoAuth=true&ctid=71c02dac-ad4f-4225-bd2d-5727e8c68a81", // GDP 차트 링크
    };
    // iframe 요소 가져오기
    const menuIframe = document.getElementById('menu-iframe');

    // 각 버튼에 클릭 이벤트 리스너 추가
    Object.keys(powerBiLinks).forEach(btnId => {
        const button = document.getElementById(btnId);
        if (button) {
            button.addEventListener('click', () => {
                // iframe의 src를 변경
                menuIframe.src = powerBiLinks[btnId];
                // 필요하다면 추가적인 스타일이나 동작 수행
                console.log(`${btnId} 클릭 - Power BI 링크 변경`);
            });
        }
    });

    // 초기 상태 설정 (기본 btn1의 PowerBI 차트 표시)
    menuIframe.src = powerBiLinks["btn1"];
});
