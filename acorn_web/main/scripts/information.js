document.addEventListener("DOMContentLoaded", () => {
    const imageSources = ["img/information-품목별 상관관계.png", "img/information-품목별 가격 편차.png", "img/information-품목별 연중 가장 저렴한 달.png"];
    let currentIndex = 0;

    const sliderImg = document.getElementById("slider-img");
    const prevBtn = document.getElementById("prev-btn");
    const nextBtn = document.getElementById("next-btn");

    // 버튼 클릭 이벤트
    prevBtn.addEventListener("click", () => {
        currentIndex = (currentIndex - 1 + imageSources.length) % imageSources.length;
        sliderImg.src = imageSources[currentIndex];
    });

    nextBtn.addEventListener("click", () => {
        currentIndex = (currentIndex + 1) % imageSources.length;
        sliderImg.src = imageSources[currentIndex];
    });
});