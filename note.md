## Embedding endpoint (/embedding)
If the input text is too long to fit in physical batch size, it will give error "input is too large to process. increase the physical batch size".
If I increase physical batch size from 512 (default value), GPU will be out of memory. The safe value for me is 512.
