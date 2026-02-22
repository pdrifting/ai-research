rem gcc vulkanTest.c -o vulkanTest -I"C:\VulkanSDK\1.4.313.1\Include" -L. -lvulkan

rem gcc vulkanTest.c -o vulkanTest -I"C:\VulkanSDK\1.4.313.1\Include" "C:\VulkanSDK\1.4.313.1\Lib\vulkan-1.lib" -static

gcc vulkanTest_patched_v3_fixed.c -o vulkanTest -I"C:\VulkanSDK\1.4.313.1\Include" "C:\VulkanSDK\1.4.313.1\Lib\vulkan-1.lib" -static