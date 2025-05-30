<template>
  <div>
    <div class="animation" />
    <div class="animation" />
  </div>
</template>

<style lang="postcss" scoped>
@import url("@knime/styles/css/mixins");

@keyframes rotate-animation {
  0% {
    transform: rotate(360deg);
  }

  100% {
    transform: rotate(0deg);
  }
}

& .animation {
  display: inline-block;
  width: 100%;
  height: 18px;
  background: linear-gradient(
    to right,
    var(--knime-porcelain) 0%,
    var(--knime-silver-sand-semi) 25%,
    var(--knime-porcelain) 50%
  );
  background-size: 200px 20px;
  background-position: -50px 0;
  animation: knight-rider 2s linear infinite;
}
</style>
