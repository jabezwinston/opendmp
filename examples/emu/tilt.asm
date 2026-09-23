; tilt.asm -- PLAN.md section 3's tilt filter, hand-compiled, in opendmp assembly (MPU-6050).
; tilt.s is the same program in InvensenseDMP's dmpasm syntax; both give the same 47 code bytes.
; This file also carries the data, so it runs with no pokes:  python -m opendmp asm tilt.asm
        .chip   mpu6050
        .entry  0x400

        .org    0x100
const:  .dw     0.0873q, 1.0q, 0.98q, 0.02q     ; GYRO_DT, 1.0, K, 1 - K
work:   .dw     0, 0, 0, 0                      ; gy, ax, t, angle (state)
param:  .dw     0.25q                           ; limit

        .org    0x400
        ; angle = K * (angle + s.gyro.y * GYRO_DT) + (1 - K) * s.accel.x
        always
        fmode   1
        p0bank  0
        p0page  0               ; P0 -> samples {ax|ay, az|temp, gx|gy, gz|-}
        p2bank  1
        p2page  1               ; P2 -> work.gy
        store
        clr                     ; gy := 0
        clr.next                ; ax := 0
        fmode   2               ; high 16-bit lane
        mov     0               ; ax.hi := accel_x
        p2page  1               ; P2 -> work.gy again
        movx    2               ; gy.hi := gyro_y (the low half, swapped up)
        fmode   1
        p0bank  1
        p0page  1               ; P0 -> work {gy, ax, t, angle}
        p1bank  1
        p1page  0               ; P1 -> const {GYRO_DT, 1.0, K, 1 - K}
        p2inc2                  ; P2 -> work.t
        mul     0,0             ; t = gy * GYRO_DT
        fmode   0
        mac     3,1             ; t = acc + angle * 1.0
        fmode   1
        mul     2,2,inc         ; P2 -> work.angle; angle = t * K (the move restarts the sum)
        fmode   0
        mac     1,3             ; angle = acc + ax * (1 - K)
        ; dmp.emit(angle, s.accel.z)
        fifo
        fmode   1
        mov     3               ; emit angle            packet[0:4]  q30
        p0bank  0
        p0page  0               ; P0 -> samples
        fmode   2
        mov     1               ; emit accel_z          packet[4:6]  i16
        ; if angle > limit: dmp.irq()
        p0bank  1
        p0page  1               ; P0 -> work
        p1page  2               ; P1 -> param
        store
        p2page  1               ; P2 -> work.gy, free by now
        fmode   1
        rsub    3,0             ; gy := limit - angle; F := limit - angle < 0 (exact: no wrap)
        if nf
        branch  join            ; angle <= limit: skip the interrupt
        always
        int
join:   always
        end
