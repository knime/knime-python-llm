import { type Ref, nextTick, onBeforeUnmount, onMounted, ref } from "vue";

export const useScrollToBottom = (container: Ref) => {
  const isAtBottom = ref(true);
  const showScrollToBottom = ref(false);

  const scrollToBottom = async () => {
    if (container.value) {
      await nextTick();
      container.value.scrollTop = container.value.scrollHeight;
    }
  };

  const handleScroll = (event: Event) => {
    const target = event.target as HTMLElement;
    const { scrollTop, scrollHeight, clientHeight } = target;

    const bottomThreshold = 100;
    isAtBottom.value =
      scrollHeight - scrollTop - clientHeight < bottomThreshold;

    showScrollToBottom.value = !isAtBottom.value;
  };

  let observer: MutationObserver;

  onMounted(() => {
    const config = { childList: true };
    observer = new MutationObserver(scrollToBottom);

    if (container.value) {
      observer.observe(container.value, config);
    }
  });

  onBeforeUnmount(() => {
    if (observer) {
      observer.disconnect();
    }
  });

  return { handleScroll };
};
