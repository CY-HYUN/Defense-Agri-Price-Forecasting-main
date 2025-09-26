// 네비게이션 버튼 클릭 시 페이지 전환 처리
document.querySelectorAll('.sidebar ul li').forEach((item) => {
    item.addEventListener('click', () => {
        const page = item.getAttribute('data-page');
        if (page) {
            window.location.href = page; // 해당 페이지로 이동
        }
    });
});

// 현재 페이지에 따라 활성 상태 적용
const currentPage = window.location.pathname.split('/').pop(); // 현재 파일명 가져오기
document.querySelectorAll('.sidebar ul li').forEach((item) => {
    const page = item.getAttribute('data-page');
    if (page === currentPage) {
        item.classList.add('active'); // 현재 페이지 버튼 활성화
    } else {
        item.classList.remove('active'); // 다른 버튼 비활성화
    }
});