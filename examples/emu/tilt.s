; tilt.s -- PLAN.md section 3's hand-compiled tilt filter (MPU-6050, entry 0x400), in dmpasm syntax.
; Data (see tilt.toml): 0x100 const {GYRO_DT, 1.0, K, 0.02}; 0x110 work {gy, ax, t, angle};
; 0x120 param {limit}.
        .lineage mantis
        always
        fmode   1
        p0bank  0
        p0page  0               ; P0 -> samples {ax|ay, az|temp, gx|gy, gz|-}
        p2bank  1
        p2page  1               ; P2 -> work.gy
        store
        .db     0xde            ; clr: gy := 0
        clr.next                ; ax := 0
        fmode   2               ; upper-half stores
        mov     0               ; ax.hi := accel_x
        p2page  1
        mov16   2               ; gy.hi := gyro_y
        fmode   1
        p0bank  1
        p0page  1               ; P0 -> work
        p1bank  1
        p1page  0               ; P1 -> const
        p2inc2                  ; P2 -> work.t
        mul     0,0             ; t = gy * GYRO_DT
        mac
        add     3,1             ; t = acc + angle * 1.0
        fmode   1
        mul     2,2,inc         ; angle = t * K
        mac
        add     1,3             ; angle = acc + ax * 0.02
        fifo
        fmode   1
        mov     3               ; emit angle
        p0bank  0
        p0page  0
        fmode   2
        mov     1               ; emit accel_z (upper half)
        p0bank  1
        p0page  1
        p1page  2               ; P1 -> param
        cmp     3,0             ; flags := angle - limit
        if le0
        branch  2
        always
        int
        always
        disable
