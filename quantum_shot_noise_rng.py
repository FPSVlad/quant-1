#!/usr/bin/env python3
"""
Демонстрация квантового shot noise фотонов и извлечение случайности с веб-камеры.

Что делает скрипт:
1) Открывает камеру (индекс 0), выставляет маленькое разрешение и целевой FPS.
2) Пытается отключить автоэкспозицию и задать короткую экспозицию.
3) 2000 раз измеряет яркость центрального ROI.
4) Строит гистограмму, считает mean/variance/sqrt(mean) для сравнения с Пуассоном.
5) Извлекает LSB-биты, делает Von Neumann debiasing.
6) Кондиционирует поток через SHA-256 для получения 256 случайных байт.
7) Печатает HEX и 8x8 матрицу из первых 64 байт.

Зависимости: стандартная библиотека + opencv-python, numpy, matplotlib.
"""

from __future__ import annotations

import hashlib
import math
import sys
from typing import List

import cv2
import matplotlib.pyplot as plt
import numpy as np


# ----------------------------- Параметры эксперимента -----------------------------
CAMERA_INDEX = 0
WIDTH = 64
HEIGHT = 48
TARGET_FPS = 30
SAMPLES = 2000
ROI_SIZE = 3  # Нечетный размер квадрата вокруг центра; 1 = ровно центральный пиксель.


# Для большинства backend OpenCV значение 1.0 отключает автоэкспозицию (manual mode).
# На некоторых ОС/драйверах семантика может отличаться — поэтому ниже проверяем, что установка прошла.
AUTO_EXPOSURE_MANUAL_VALUE = 1.0
# Небольшая (короткая) экспозиция уменьшает усреднение фотонных флуктуаций.
EXPOSURE_VALUE = -6.0


# Нужны 256 байт = 2048 бит. Каждый SHA-256 блок даёт 32 байта.
# Значит, достаточно 8 хешей из разных чанков debiased-данных.
OUTPUT_BYTES = 256
HASH_BLOCK_BYTES = 32
HASHES_NEEDED = OUTPUT_BYTES // HASH_BLOCK_BYTES


def open_camera() -> cv2.VideoCapture:
    """Открывает веб-камеру и выставляет параметры захвата."""
    cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        raise RuntimeError("Не удалось открыть веб-камеру (индекс 0).")

    # Маленькое разрешение и целевой FPS повышают стабильность и скорость набора выборки.
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, HEIGHT)
    cap.set(cv2.CAP_PROP_FPS, TARGET_FPS)

    # Пытаемся отключить автоэкспозицию и включить ручную короткую экспозицию.
    cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, AUTO_EXPOSURE_MANUAL_VALUE)
    cap.set(cv2.CAP_PROP_EXPOSURE, EXPOSURE_VALUE)

    # Прогреваем камеру, чтобы автонастройки/буфер стабилизировались.
    for _ in range(10):
        cap.read()

    # Вывод фактических параметров, чтобы пользователь видел, что реально применилось драйвером.
    actual_w = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    actual_h = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    actual_fps = cap.get(cv2.CAP_PROP_FPS)
    actual_auto_exp = cap.get(cv2.CAP_PROP_AUTO_EXPOSURE)
    actual_exp = cap.get(cv2.CAP_PROP_EXPOSURE)

    print("Параметры камеры (фактические):")
    print(f"  Разрешение: {actual_w:.0f}x{actual_h:.0f}")
    print(f"  FPS: {actual_fps:.2f}")
    print(f"  AUTO_EXPOSURE: {actual_auto_exp}")
    print(f"  EXPOSURE: {actual_exp}")
    print()

    return cap


def central_roi_intensity(frame_bgr: np.ndarray, roi_size: int) -> int:
    """
    Возвращает яркость центрального ROI в диапазоне 0..255.

    Мы берём серый канал (интенсивность) и усредняем маленький центральный регион.
    Такой сигнал содержит флуктуации числа пришедших фотонов (shot noise),
    которые в первом приближении подчиняются статистике Пуассона.
    """
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape

    half = roi_size // 2
    cy, cx = h // 2, w // 2
    y1, y2 = max(0, cy - half), min(h, cy + half + 1)
    x1, x2 = max(0, cx - half), min(w, cx + half + 1)

    roi = gray[y1:y2, x1:x2]
    return int(np.round(float(np.mean(roi))))


def sample_brightness(cap: cv2.VideoCapture, samples: int, roi_size: int) -> np.ndarray:
    """Снимает N измерений яркости центрального ROI."""
    values: List[int] = []

    while len(values) < samples:
        ok, frame = cap.read()
        if not ok:
            continue
        values.append(central_roi_intensity(frame, roi_size))

    return np.array(values, dtype=np.uint8)


def von_neumann_debias(bits: List[int]) -> List[int]:
    """
    Von Neumann debiasing:
    - пары 01 -> 0
    - пары 10 -> 1
    - пары 00 и 11 отбрасываются

    Это убирает смещение, если биты независимы и имеют постоянную вероятность 0/1.
    """
    out: List[int] = []
    usable = len(bits) - (len(bits) % 2)
    for i in range(0, usable, 2):
        b1, b2 = bits[i], bits[i + 1]
        if b1 == 0 and b2 == 1:
            out.append(0)
        elif b1 == 1 and b2 == 0:
            out.append(1)
    return out


def bits_to_bytes(bits: List[int]) -> bytes:
    """Упаковывает список битов (0/1) в байты (MSB first внутри байта)."""
    usable = len(bits) - (len(bits) % 8)
    bits = bits[:usable]

    out = bytearray()
    for i in range(0, usable, 8):
        byte = 0
        for b in bits[i : i + 8]:
            byte = (byte << 1) | b
        out.append(byte)
    return bytes(out)


def sha256_conditioner(debiased_bits: List[int], output_bytes: int = OUTPUT_BYTES) -> bytes:
    """
    Кондиционирование источника случайности через SHA-256.

    Мы берём последовательные чанки debiased-данных и хешируем их.
    Конкатенация дайджестов даёт криптографически стойкое «сжатие» энтропии.
    """
    if output_bytes % 32 != 0:
        raise ValueError("output_bytes должен быть кратен 32 для SHA-256 блоков.")

    need_hashes = output_bytes // 32
    raw = bits_to_bytes(debiased_bits)

    # Берем равные чанки исходных байтов для независимых хешей.
    # Минимум 32 байта на хеш (можно больше; здесь выбираем динамически).
    min_total = need_hashes * 32
    if len(raw) < min_total:
        raise RuntimeError(
            f"Недостаточно debiased-данных для SHA-256 conditioning: "
            f"нужно >= {min_total} байт, получено {len(raw)}"
        )

    chunk_size = len(raw) // need_hashes
    conditioned = bytearray()

    for i in range(need_hashes):
        start = i * chunk_size
        end = (i + 1) * chunk_size if i < need_hashes - 1 else len(raw)
        chunk = raw[start:end]
        digest = hashlib.sha256(chunk).digest()
        conditioned.extend(digest)

    return bytes(conditioned[:output_bytes])


def main() -> int:
    cap = open_camera()
    try:
        brightness = sample_brightness(cap, SAMPLES, ROI_SIZE)
    finally:
        cap.release()

    # Статистика яркости: для чистого Пуассона var ~ mean, а sigma ~ sqrt(mean).
    mean_val = float(np.mean(brightness))
    var_val = float(np.var(brightness, ddof=0))
    sqrt_mean = math.sqrt(mean_val)

    print(f"Собрано измерений: {len(brightness)}")
    print(f"Средняя яркость: {mean_val:.4f}")
    print(f"Дисперсия: {var_val:.4f}")
    print(f"sqrt(среднего): {sqrt_mean:.4f}")
    print()

    # Извлекаем младший бит каждого измерения как «сырой» источник случайности.
    raw_bits = [int(v & 1) for v in brightness]
    debiased_bits = von_neumann_debias(raw_bits)

    print(f"Сырых LSB-битов: {len(raw_bits)}")
    print(f"После Von Neumann debiasing: {len(debiased_bits)} бит")
    print()

    try:
        random_bytes = sha256_conditioner(debiased_bits, OUTPUT_BYTES)
    except RuntimeError as exc:
        print("Ошибка: недостаточно энтропии после debiasing для 256 байт.")
        print(str(exc))
        print("Попробуйте увеличить SAMPLES или ROI_SIZE=1 для меньшей корреляции.")
        return 1

    print("256 квантовых случайных байтов (HEX):")
    print(random_bytes.hex())
    print()

    # Первые 64 байта превращаем в матрицу 8x8 для вычислительных задач.
    matrix8x8 = np.frombuffer(random_bytes[:64], dtype=np.uint8).reshape((8, 8))
    print("Случайная матрица 8x8 из первых 64 байт:")
    print(matrix8x8)

    # Гистограмма яркости демонстрирует распределение флуктуаций интенсивности.
    plt.figure(figsize=(8, 4))
    plt.hist(brightness, bins=32, range=(0, 255), color="royalblue", alpha=0.85)
    plt.title("Гистограмма яркости центрального ROI (shot noise)")
    plt.xlabel("Яркость (0..255)")
    plt.ylabel("Частота")
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.show()

    return 0


if __name__ == "__main__":
    sys.exit(main())
