# Fast Face Swapping with High-Fidelity Lightweight Generator Assisted by Online Knowledge Distillation (TVC 2024)

This is the official code for "Fast Face Swapping with High-Fidelity Lightweight Generator Assisted by Online Knowledge Distillation", accepted by The Visual Computer 2024.

Paper link: https://link.springer.com/article/10.1007/s00371-024-03414-2

# Abstract
Advanced face swapping approaches have achieved high-fidelity results. However, the success of most methods hinges on heavy parameters and high-computational costs. With the popularity of real-time face swapping, these factors have become obstacles restricting their swap speed and application. To overcome these challenges, we propose a high-fidelity lightweight generator (HFLG) for face swapping, which is a compressed version of the existing network Simple Swap and consists of its 1/4 channels. Moreover, to stabilize the learning of HFLG, we introduce feature map-based online knowledge distillation into our training process and improve the teacher–student architecture. Specifically, we first enhance our teacher generator to provide more efficient guidance. It minimizes the loss of details on the lower face. In addition, a new identity-irrelevant similarity loss is proposed to improve the preservation of non-facial regions in the teacher generator results. Furthermore, HFLG uses an extended identity injection module to inject identity more efficiently. It gradually learns face swapping by imitating the feature maps and outputs of the teacher generator online. Extensive experiments on faces in the wild demonstrate that our method achieves comparable results with other methods while having fewer parameters, lower computations, and faster inference speed.

![Fig 1](https://github.com/EifelTing/HFLFS/blob/main/Fig%201.svg)
