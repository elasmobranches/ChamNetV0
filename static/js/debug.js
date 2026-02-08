// Debug.js - 노출 디버깅 웹 인터페이스 제어

// 상태 업데이트 주기 (ms)
const STATUS_UPDATE_INTERVAL = 500;

// 현재 모드 ('auto' or 'manual')
let currentMode = 'auto';

// 상태 업데이트 함수
async function updateStatus() {
    try {
        const response = await fetch('/api/status');
        const data = await response.json();

        // 모드 표시
        const modeIndicator = document.getElementById('modeIndicator');
        const toggleBtn = document.getElementById('toggleBtn');
        const exposureSlider = document.getElementById('exposureSlider');
        const gainSlider = document.getElementById('gainSlider');

        if (data.mode === 'auto') {
            modeIndicator.textContent = 'AUTO EXPOSURE';
            modeIndicator.className = 'mode-indicator mode-auto';
            toggleBtn.textContent = '🔄 Manual 모드로 전환';
            exposureSlider.disabled = true;
            gainSlider.disabled = true;
            currentMode = 'auto';
        } else {
            modeIndicator.textContent = 'MANUAL EXPOSURE';
            modeIndicator.className = 'mode-indicator mode-manual';
            toggleBtn.textContent = '🔄 Auto 모드로 전환';
            exposureSlider.disabled = false;
            gainSlider.disabled = false;
            currentMode = 'manual';
        }

        // 현재 노출/게인 값
        if (data.exposure !== undefined && data.exposure !== null) {
            document.getElementById('currentExposure').textContent = data.exposure;
        }

        if (data.gain !== undefined && data.gain !== null) {
            document.getElementById('currentGain').textContent = data.gain;
        }

        // Manual 모드일 때만 슬라이더 값 동기화 (API에서 받은 값으로)
        if (currentMode === 'manual') {
            if (data.manual_exposure !== undefined && data.manual_exposure !== null) {
                exposureSlider.value = data.manual_exposure;
                document.getElementById('exposureValue').textContent = data.manual_exposure;
            }
            if (data.manual_gain !== undefined && data.manual_gain !== null) {
                gainSlider.value = data.manual_gain;
                document.getElementById('gainValue').textContent = data.manual_gain;
            }
        }

        // 밝기 통계
        if (data.brightness_mean !== undefined && data.brightness_mean !== null) {
            document.getElementById('brightnessMean').textContent = data.brightness_mean.toFixed(1);
        }

        if (data.brightness_avg !== undefined && data.brightness_avg !== null) {
            document.getElementById('brightnessAvg').textContent = data.brightness_avg.toFixed(1);
        }

        if (data.brightness_std !== undefined && data.brightness_std !== null) {
            document.getElementById('brightnessStd').textContent = data.brightness_std.toFixed(1);
        }

        // 마커 검출 통계
        if (data.frame_count !== undefined && data.frame_count !== null) {
            document.getElementById('frameCount').textContent = data.frame_count;
        }

        if (data.detection_count !== undefined && data.detection_count !== null) {
            document.getElementById('detectionCount').textContent = data.detection_count;
        }

        if (data.detection_rate !== undefined && data.detection_rate !== null) {
            document.getElementById('detectionRate').textContent = `${data.detection_rate.toFixed(1)}%`;
        }

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

// Auto/Manual 모드 토글
async function toggleAutoExposure() {
    try {
        const response = await fetch('/api/toggle_auto', {
            method: 'POST'
        });

        const data = await response.json();

        if (data.success) {
            console.log(`모드 변경: ${data.mode}`);
            // 즉시 상태 업데이트
            updateStatus();
        } else {
            alert(`오류: ${data.error}`);
        }
    } catch (error) {
        console.error('모드 전환 실패:', error);
        alert('모드 전환에 실패했습니다.');
    }
}

// 노출 값 설정
async function setExposure(value) {
    try {
        const response = await fetch('/api/set_exposure', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ exposure: parseInt(value) })
        });

        const data = await response.json();

        if (data.success) {
            console.log(`노출 설정: ${data.exposure}`);
        } else {
            alert(`오류: ${data.error}`);
        }
    } catch (error) {
        console.error('노출 설정 실패:', error);
        alert('노출 설정에 실패했습니다.');
    }
}

// 게인 값 설정
async function setGain(value) {
    try {
        const response = await fetch('/api/set_gain', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ gain: parseInt(value) })
        });

        const data = await response.json();

        if (data.success) {
            console.log(`게인 설정: ${data.gain}`);
        } else {
            alert(`오류: ${data.error}`);
        }
    } catch (error) {
        console.error('게인 설정 실패:', error);
        alert('게인 설정에 실패했습니다.');
    }
}

// 밝기 값 설정
async function setBrightness(value) {
    try {
        const response = await fetch('/api/set_brightness', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ brightness: parseInt(value) })
        });

        const data = await response.json();

        if (data.success) {
            console.log(`밝기 설정: ${data.brightness}`);
        } else {
            alert(`오류: ${data.error}`);
        }
    } catch (error) {
        console.error('밝기 설정 실패:', error);
        alert('밝기 설정에 실패했습니다.');
    }
}

// 대비 값 설정
async function setContrast(value) {
    try {
        const response = await fetch('/api/set_contrast', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ contrast: parseInt(value) })
        });

        const data = await response.json();

        if (data.success) {
            console.log(`대비 설정: ${data.contrast}`);
        } else {
            alert(`오류: ${data.error}`);
        }
    } catch (error) {
        console.error('대비 설정 실패:', error);
        alert('대비 설정에 실패했습니다.');
    }
}

// 선명도 값 설정
async function setSharpness(value) {
    try {
        const response = await fetch('/api/set_sharpness', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ sharpness: parseInt(value) })
        });

        const data = await response.json();

        if (data.success) {
            console.log(`선명도 설정: ${data.sharpness}`);
        } else {
            alert(`오류: ${data.error}`);
        }
    } catch (error) {
        console.error('선명도 설정 실패:', error);
        alert('선명도 설정에 실패했습니다.');
    }
}

// 리셋
async function resetValues() {
    try {
        const response = await fetch('/api/reset', {
            method: 'POST'
        });

        const data = await response.json();

        if (data.success) {
            console.log('리셋 완료: exposure=50, gain=50, brightness=4, contrast=4, sharpness=4');

            // 슬라이더 값 업데이트
            document.getElementById('exposureSlider').value = data.exposure;
            document.getElementById('exposureValue').textContent = data.exposure;
            document.getElementById('gainSlider').value = data.gain;
            document.getElementById('gainValue').textContent = data.gain;
            document.getElementById('brightnessSlider').value = data.brightness;
            document.getElementById('brightnessValue').textContent = data.brightness;
            document.getElementById('contrastSlider').value = data.contrast;
            document.getElementById('contrastValue').textContent = data.contrast;
            document.getElementById('sharpnessSlider').value = data.sharpness;
            document.getElementById('sharpnessValue').textContent = data.sharpness;
        } else {
            alert(`오류: ${data.error}`);
        }
    } catch (error) {
        console.error('리셋 실패:', error);
        alert('리셋에 실패했습니다.');
    }
}

// 설정 저장
async function saveSettings() {
    try {
        const response = await fetch('/api/save_settings', {
            method: 'POST'
        });

        const data = await response.json();

        if (data.success) {
            alert('✅ 현재 설정이 로그에 저장되었습니다.\n\n터미널 로그를 확인하세요.');
        } else {
            alert(`오류: ${data.error}`);
        }
    } catch (error) {
        console.error('설정 저장 실패:', error);
        alert('설정 저장에 실패했습니다.');
    }
}

// 종료
async function shutdown() {
    if (!confirm('정말 종료하시겠습니까?')) {
        return;
    }

    try {
        const response = await fetch('/api/shutdown', {
            method: 'POST'
        });

        const data = await response.json();

        if (data.success) {
            alert('✅ 프로그램이 종료됩니다.\n\n창을 닫아도 됩니다.');

            // 페이지를 종료 메시지로 변경
            document.body.innerHTML = `
                <div style="display: flex; flex-direction: column; align-items: center; justify-content: center;
                            height: 100vh; background: linear-gradient(135deg, #1e3a8a 0%, #1e293b 100%);
                            color: white; font-family: sans-serif;">
                    <h1 style="font-size: 3rem; margin-bottom: 20px;">✅ 종료 완료</h1>
                    <p style="font-size: 1.5rem; margin-bottom: 10px;">디버깅 세션이 종료되었습니다.</p>
                    <p style="font-size: 1rem; color: #6b7280; margin-top: 30px;">이 창을 닫아도 됩니다.</p>
                </div>
            `;
        }
    } catch (error) {
        console.error('종료 요청 실패:', error);
        alert('✅ 서버가 종료되었습니다.\n\n창을 닫아도 됩니다.');
    }
}

// 초기화 및 이벤트 리스너
document.addEventListener('DOMContentLoaded', () => {
    console.log('ZED 노출 디버깅 도구 시작');

    // 슬라이더 이벤트 리스너
    const exposureSlider = document.getElementById('exposureSlider');
    const gainSlider = document.getElementById('gainSlider');
    const brightnessSlider = document.getElementById('brightnessSlider');
    const contrastSlider = document.getElementById('contrastSlider');
    const sharpnessSlider = document.getElementById('sharpnessSlider');

    // 노출 슬라이더
    exposureSlider.addEventListener('input', (e) => {
        const value = e.target.value;
        document.getElementById('exposureValue').textContent = value;
    });

    exposureSlider.addEventListener('change', (e) => {
        const value = e.target.value;
        setExposure(value);
    });

    // 게인 슬라이더
    gainSlider.addEventListener('input', (e) => {
        const value = e.target.value;
        document.getElementById('gainValue').textContent = value;
    });

    gainSlider.addEventListener('change', (e) => {
        const value = e.target.value;
        setGain(value);
    });

    // 밝기 슬라이더
    brightnessSlider.addEventListener('input', (e) => {
        const value = e.target.value;
        document.getElementById('brightnessValue').textContent = value;
    });

    brightnessSlider.addEventListener('change', (e) => {
        const value = e.target.value;
        setBrightness(value);
    });

    // 대비 슬라이더
    contrastSlider.addEventListener('input', (e) => {
        const value = e.target.value;
        document.getElementById('contrastValue').textContent = value;
    });

    contrastSlider.addEventListener('change', (e) => {
        const value = e.target.value;
        setContrast(value);
    });

    // 선명도 슬라이더
    sharpnessSlider.addEventListener('input', (e) => {
        const value = e.target.value;
        document.getElementById('sharpnessValue').textContent = value;
    });

    sharpnessSlider.addEventListener('change', (e) => {
        const value = e.target.value;
        setSharpness(value);
    });

    // 초기 상태 업데이트
    updateStatus();

    // 주기적으로 상태 업데이트
    setInterval(updateStatus, STATUS_UPDATE_INTERVAL);
});
