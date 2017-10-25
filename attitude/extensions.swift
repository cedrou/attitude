//
//  extensions.swift
//  attitude
//
//  Created by Cédric Rousseau on 24/10/2017.
//  Copyright © 2017 anamnes. All rights reserved.
//

import Foundation
import CoreMotion


extension simd_quatf {
    init(_ q:CMQuaternion) {
        self.init(ix:Float(q.x), iy:Float(q.y), iz:Float(q.z), r:Float(q.w))
    }
}

extension simd_float4x4 {
    init(_ q:CMQuaternion) {
        self.init(simd_quatf(q))
    }
}

extension simd_float3 {
    init(_ a:CMAcceleration) {
        self.init(Float(a.x), Float(a.y), Float(a.z))
    }
    
    func norm2() -> Float {
        return x*x + y*y + z*z
    }
}
