// Stream.js - 웹 인터페이스 제어

// 상태 업데이트 주기 (ms)
const STATUS_UPDATE_INTERVAL = 500;

// 상태 업데이트 함수
async function updateStatus() {
    try {
        const response = await fetch('/api/status');
        const data = await response.json();

        // 거리 표시
        if (data.current_distance !== null) {
            document.getElementById('currentDistance').textContent =
                `${data.current_distance.toFixed(3)} m`;
        } else {
            document.getElementById('currentDistance').textContent = '-';
        }

        // ZED Depth 표시
        if (data.zed_depth !== null) {
            document.getElementById('zedDepth').textContent =
                `${(data.zed_depth / 1000).toFixed(3)} m`;
        } else {
            document.getElementById('zedDepth').textContent = '-';
        }

        // 오차율 표시
        if (data.error_pct !== null && !isNaN(data.error_pct)) {
            const errorElement = document.getElementById('errorPct');
            errorElement.textContent = `${data.error_pct.toFixed(2)} %`;

            // 오차율에 따른 색상 변경
            errorElement.className = 'value';
            if (data.error_pct < 2.0) {
                errorElement.classList.add('error-low');
            } else if (data.error_pct < 5.0) {
                errorElement.classList.add('error-medium');
            } else {
                errorElement.classList.add('error-high');
            }
        } else {
            document.getElementById('errorPct').textContent = '-';
        }

        // 저장된 프레임 수
        document.getElementById('savedCount').textContent = data.saved_count || 0;

        // 측정 모드
        const modeText = data.mode === 'marker_region' ? '마커 영역' : '윈도우';
        document.getElementById('currentMode').textContent = modeText;

        // 마커 감지 상태
        const indicator = document.getElementById('markerIndicator');
        const markerText = document.getElementById('markerText');

        if (data.marker_detected) {
            indicator.className = 'status-indicator active';
            markerText.textContent = '마커 감지됨';
        } else {
            indicator.className = 'status-indicator inactive';
            markerText.textContent = '마커 감지 대기 중...';
        }

    } catch (error) {
        console.error('상태 업데이트 실패:', error);
    }
}

// 거리 설정
async function setDistance() {
    const input = document.getElementById('distanceInput');
    const distance = parseFloat(input.value);

    if (isNaN(distance) || distance <= 0) {
        alert('올바른 거리를 입력하세요 (0보다 큰 값)');
        return;
    }

    try {
        const response = await fetch('/api/set_distance', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ distance: distance })
        });

        const data = await response.json();

        if (data.success) {
            console.log(`거리 설정: ${distance}m`);
            input.value = ''; // 입력란 초기화
        } else {
            alert(`오류: ${data.error}`);
        }
    } catch (error) {
        console.error('거리 설정 실패:', error);
        alert('거리 설정에 실패했습니다.');
    }
}

// 프레임 저장
async function saveFrame(event) {
    try {
        const response = await fetch('/api/save_frame', {
            method: 'POST'
        });

        // HTTP 상태 코드 확인
        if (!response.ok) {
            const data = await response.json();
            throw new Error(data.error || `HTTP ${response.status}: ${response.statusText}`);
        }

        const data = await response.json();

        if (data.success) {
            console.log(`✓ 프레임 저장 완료 (총 ${data.saved_count}개)`);

            // 저장 성공 피드백 (버튼 변경)
            if (event && event.target) {
                const btn = event.target;
                const originalText = btn.textContent;
                btn.textContent = '✓ 저장됨';
                setTimeout(() => {
                    btn.textContent = originalText;
                }, 1500);
            }

            // 성공 메시지 표시 (선택적)
            showNotification(`프레임 저장 완료! (총 ${data.saved_count}개)`, 'success');
        } else {
            // 서버에서 success: false를 반환한 경우
            throw new Error(data.error || '알 수 없는 오류');
        }
    } catch (error) {
        console.error('❌ 프레임 저장 실패:', error);

        // 에러 메시지 파싱
        let errorMessage = error.message || '알 수 없는 오류';

        // 사용자 친화적인 에러 메시지 표시
        if (errorMessage.includes('거리가 설정되지')) {
            alert(`❌ 거리를 먼저 설정하세요!\n\n1. 거리 입력란에 값 입력 (예: 2.0)\n2. "설정" 버튼 클릭\n3. 다시 프레임 저장`);
        } else if (errorMessage.includes('마커가 감지되지')) {
            alert(`❌ 마커가 감지되지 않았습니다!\n\n1. 마커를 70cm 이상 거리에 배치\n2. 마커가 화면에 보이는지 확인\n3. "Marker: Detected" 표시 확인`);
        } else {
            alert(`❌ 프레임 저장 실패\n\n오류: ${errorMessage}\n\n해결 방법:\n- 거리가 설정되었는지 확인 (D 키)\n- 마커가 감지되었는지 확인`);
        }
    }
}

// 알림 표시 함수
function showNotification(message, type = 'info') {
    console.log(`[${type.toUpperCase()}] ${message}`);

    // TODO: 나중에 토스트 알림으로 개선 가능
    // 예: 화면 우측 상단에 알림 표시
}

// 모드 변경
async function toggleMode() {
    try {
        const response = await fetch('/api/toggle_mode', {
            method: 'POST'
        });

        const data = await response.json();

        if (data.success) {
            const modeText = data.mode === 'marker_region' ? '마커 영역' : '윈도우';
            console.log(`모드 변경: ${modeText}`);
        } else {
            alert(`오류: ${data.error}`);
        }
    } catch (error) {
        console.error('모드 변경 실패:', error);
        alert('모드 변경에 실패했습니다.');
    }
}

// CSV 다운로드
async function downloadCSV() {
    try {
        const response = await fetch('/api/download_csv', {
            method: 'POST'
        });

        const data = await response.json();

        if (data.success) {
            alert('CSV 파일이 서버에 저장되었습니다.');
        } else {
            alert(`오류: ${data.error}`);
        }
    } catch (error) {
        console.error('CSV 다운로드 실패:', error);
        alert('CSV 다운로드에 실패했습니다.');
    }
}

// 종료
async function shutdown() {
    if (!confirm('정말 종료하시겠습니까?\n\n측정 기록이 자동으로 CSV 파일로 저장됩니다.')) {
        return;
    }

    try {
        const response = await fetch('/api/shutdown', {
            method: 'POST'
        });

        const data = await response.json();

        if (data.success) {
            // 종료 메시지 표시
            alert('✅ 프로그램이 종료됩니다.\n\n📁 측정 기록이 CSV 파일로 저장되었습니다.\n📍 위치: data/results/\n\n창을 닫아도 됩니다.');

            // 페이지를 종료 메시지로 변경
            document.body.innerHTML = `
                <div style="display: flex; flex-direction: column; align-items: center; justify-content: center;
                            height: 100vh; background: linear-gradient(135deg, #1e3a8a 0%, #1e293b 100%);
                            color: white; font-family: sans-serif;">
                    <h1 style="font-size: 3rem; margin-bottom: 20px;">✅ 종료 완료</h1>
                    <p style="font-size: 1.5rem; margin-bottom: 10px;">측정 기록이 CSV 파일로 저장되었습니다.</p>
                    <p style="font-size: 1.2rem; color: #9ca3af;">📁 위치: data/results/</p>
                    <p style="font-size: 1rem; color: #6b7280; margin-top: 30px;">이 창을 닫아도 됩니다.</p>
                </div>
            `;
        }
    } catch (error) {
        console.error('종료 요청 실패:', error);
        // 네트워크 오류는 정상 (서버가 종료되었기 때문)
        alert('✅ 서버가 종료되었습니다.\n\n창을 닫아도 됩니다.');
    }
}

// 키보드 단축키
document.addEventListener('keydown', (event) => {
    // 입력란에 포커스되어 있으면 무시
    if (event.target.tagName === 'INPUT') {
        if (event.key === 'Enter') {
            setDistance();
        }
        return;
    }

    switch (event.key.toLowerCase()) {
        case 'd':
            document.getElementById('distanceInput').focus();
            break;
        case 's':
            saveFrame();
            break;
        case 'm':
            toggleMode();
            break;
        case 'q':
            shutdown();
            break;
    }
});

// 초기화 및 주기적 상태 업데이트
document.addEventListener('DOMContentLoaded', () => {
    console.log('ZED ArUco Depth Estimation 시작');

    // 초기 상태 업데이트
    updateStatus();

    // 주기적으로 상태 업데이트
    setInterval(updateStatus, STATUS_UPDATE_INTERVAL);
});
