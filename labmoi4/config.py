IMAGE_W = 1920
IMAGE_H = 1080
ASPECT = float(IMAGE_W) / float(IMAGE_H)
MAX_SPP = 4096
MAX_TRACE_DEPTH = 1024
SPHERE_COUNT = 14

# Джойнт билатераль (2 прохода: по X и по Y). Больше radius / sig_* — сильнее сглаживание шума.
BILATERAL_RADIUS = 3
# Пространственное σ в пикселях (Gaussian по смещению).
BILATERAL_SIG_S = 2.35
# Range в log(яркость): больше значение — больше смешивание при шумном MC (раньше ~0.28 почти «отключало» фильтр).
BILATERAL_SIG_R_LOG = 0.82
# Чувствительность к разнице глубины / нормалей (выше — легче усреднять внутри объекта).
BILATERAL_SIG_Z = 1.05
BILATERAL_SIG_N = 0.48
# Вес при разном id объекта (0..1); меньше — жёстче границы.
BILATERAL_DIFF_OBJ_WEIGHT = 0.07
