<template>
  <span class="animated-ellipsis">
    <span class="dot dot-1">.</span>
    <span class="dot dot-2">.</span>
    <span class="dot dot-3">.</span>
  </span>
</template>

<style scoped>
.animated-ellipsis {
  display: inline-block;
}

.dot {
  display: inline-block;
  animation: wave 1.4s ease-in-out infinite both;
}

.dot-1 {
  animation-delay: -0.32s;
}

.dot-2 {
  animation-delay: -0.16s;
}

.dot-3 {
  animation-delay: 0s;
}

@keyframes wave {
  0%,
  80%,
  100% {
    transform: translateY(0);
  }

  40% {
    transform: translateY(-3px);
  }
}
</style>
