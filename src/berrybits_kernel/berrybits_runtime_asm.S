.text
.globl berrybits_runtime_q4_dot_asm
.type berrybits_runtime_q4_dot_asm, @function

berrybits_runtime_q4_dot_asm:
    xorps %xmm0, %xmm0
    xorq %rax, %rax

.loop:
    cmpq %rdx, %rax
    jae .done

    movss (%rdi, %rax, 4), %xmm1
    mulss (%rsi, %rax, 4), %xmm1
    addss %xmm1, %xmm0
    incq %rax
    jmp .loop

.done:
    ret

.size berrybits_runtime_q4_dot_asm, .-berrybits_runtime_q4_dot_asm
