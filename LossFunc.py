import torch
import torch.nn as nn
import torch.nn.functional as func

class MyLoss(nn.Module):
    def __init__(self):
        super(MyLoss, self).__init__()
        return

    def forward(self, pred, trut):
        dim = pred.shape 
        # dim[0]=sample length, dim[1]=3 variable p,u,v
        # dim[2]=x length=128, dim[3]=y length=128
        u = 0
        v = 1
        # dx = 1. # TODO or 2./dim[2] ? the length between two points
        # dy = 1. # or 2./dim[3] ?
        iRe = 1.0/400 # =1/Re=nu/(V*L)=nu=1e-5 have divided rho and V
    
        x = 2
        y = 3
        ixm = torch.arange(1, dim[x]-1).cuda() # index x-row middle
        ixl = torch.arange(0, dim[x]-2).cuda() # left
        ixr = torch.arange(2, dim[x]).cuda()   # right
        iym = torch.arange(1, dim[y]-1).cuda() # index y-row middle
        iyb = torch.arange(0, dim[y]-2).cuda() # bottom
        iyt = torch.arange(2, dim[y]).cuda()   # top

        mid0 = torch.index_select(trut, x, ixm)
        mid  = torch.index_select(mid0, y, iym)
        lef0 = torch.index_select(trut, x, ixl)
        lef  = torch.index_select(lef0, y, iym)
        rig0 = torch.index_select(trut, x, ixr)
        rig  = torch.index_select(rig0, y, iym)
        bot0 = torch.index_select(trut, x, ixm)
        bot  = torch.index_select(bot0, y, iyb)
        top0 = torch.index_select(trut, x, ixm)
        top  = torch.index_select(top0, y, iyt)

        # ip = torch.LongTensor([p]).cuda()
        iu = torch.LongTensor([u]).cuda()
        iv = torch.LongTensor([v]).cuda()

        # pm0 = torch.index_select(mid, 1, ip)
        um0 = torch.index_select(mid, 1, iu)
        vm0 = torch.index_select(mid, 1, iv)
        # pl0 = torch.index_select(lef, 1, ip)
        ul0 = torch.index_select(lef, 1, iu)
        vl0 = torch.index_select(lef, 1, iv)
        # pr0 = torch.index_select(rig, 1, ip)
        ur0 = torch.index_select(rig, 1, iu)
        vr0 = torch.index_select(rig, 1, iv)
        # pb0 = torch.index_select(bot, 1, ip)
        ub0 = torch.index_select(bot, 1, iu)
        vb0 = torch.index_select(bot, 1, iv)
        # pt0 = torch.index_select(top, 1, ip)
        ut0 = torch.index_select(top, 1, iu)
        vt0 = torch.index_select(top, 1, iv)

        # pm = torch.squeeze(pm0) # delete dimension 1, which length=1
        um = torch.squeeze(um0)
        vm = torch.squeeze(vm0)
        # pl = torch.squeeze(pl0)
        ul = torch.squeeze(ul0)
        vl = torch.squeeze(vl0)
        # pr = torch.squeeze(pr0)
        ur = torch.squeeze(ur0)
        vr = torch.squeeze(vr0)
        # pb = torch.squeeze(pb0)
        ub = torch.squeeze(ub0)
        vb = torch.squeeze(vb0)
        # pt = torch.squeeze(pt0)
        ut = torch.squeeze(ut0)
        vt = torch.squeeze(vt0)

        # for pred data
        mid0_ = torch.index_select(pred,  x, ixm)
        mid_  = torch.index_select(mid0_, y, iym)
        lef0_ = torch.index_select(pred,  x, ixl)
        lef_  = torch.index_select(lef0_, y, iym)
        rig0_ = torch.index_select(pred,  x, ixr)
        rig_  = torch.index_select(rig0_, y, iym)
        bot0_ = torch.index_select(pred,  x, ixm)
        bot_  = torch.index_select(bot0_, y, iyb)
        top0_ = torch.index_select(pred,  x, ixm)
        top_  = torch.index_select(top0_, y, iyt)

        # pm0_ = torch.index_select(mid_, 1, ip)
        um0_ = torch.index_select(mid_, 1, iu)
        vm0_ = torch.index_select(mid_, 1, iv)
        # pl0_ = torch.index_select(lef_, 1, ip)
        ul0_ = torch.index_select(lef_, 1, iu)
        vl0_ = torch.index_select(lef_, 1, iv)
        # pr0_ = torch.index_select(rig_, 1, ip)
        ur0_ = torch.index_select(rig_, 1, iu)
        vr0_ = torch.index_select(rig_, 1, iv)
        # pb0_ = torch.index_select(bot_, 1, ip)
        ub0_ = torch.index_select(bot_, 1, iu)
        vb0_ = torch.index_select(bot_, 1, iv)
        # pt0_ = torch.index_select(top_, 1, ip)
        ut0_ = torch.index_select(top_, 1, iu)
        vt0_ = torch.index_select(top_, 1, iv)

        # pm_ = torch.squeeze(pm0_)
        um_ = torch.squeeze(um0_)
        vm_ = torch.squeeze(vm0_)
        # pl_ = torch.squeeze(pl0_)
        ul_ = torch.squeeze(ul0_)
        vl_ = torch.squeeze(vl0_)
        # pr_ = torch.squeeze(pr0_)
        ur_ = torch.squeeze(ur0_)
        vr_ = torch.squeeze(vr0_)
        # pb_ = torch.squeeze(pb0_)
        ub_ = torch.squeeze(ub0_)
        vb_ = torch.squeeze(vb0_)
        # pt_ = torch.squeeze(pt0_)
        ut_ = torch.squeeze(ut0_)
        vt_ = torch.squeeze(vt0_)

        # (for incompressible flow, rho is constant)
        # mass conservation
        # du/dx + dv/dy = 0   ==>
        # |(du/dx + dv/dy) - (du'/dx + dv'/dy)|
        dudx  = (ur  - ul ) * 0.5
        dvdy  = (vt  - vb ) * 0.5
        du_dx = (ur_ - ul_) * 0.5
        dv_dy = (vt_ - vb_) * 0.5
        # TODO or use abs for each item of the conservation law
        lossMass0 = torch.abs((dudx + dvdy) - (du_dx + dv_dy))
        lossMass = torch.mean(lossMass0)

        # momentum conservation
        # x direction
        # d(u^2)/dx + d(uv)/dy + dp/dx - d(Txx)/dx - d(Tyx)/dy = 0   ==>
        # |[d(u^2)/dx + d(uv)/dy + dp/dx - d(Txx)/dx - d(Tyx)/dy] -
        #  [d(u'^2)/dx + d(u'v')/dy + dp'/dx - d(Txx')/dx - d(Tyx')/dy]|
        # y direction
        # d(vu)/dx + d(v^2)/dy + dp/dy - d(Txy)/dx - d(Tyy)/dy = 0   ==>
        # |[d(vu)/dx + d(v^2)/dy + dp/dy - d(Txy)/dx - d(Tyy)/dy] -
        # [d(v'u')/dx + d(v'^2)/dy + dp'/dy - d(Txy')/dx - d(Tyy')/dy]|
        du2dx = 0.25 * ( (ur + um) ** 2 - (ul + um) ** 2 )
        duvdy = 0.25 * ( (ut + um) * (vt + vm) - (ub + um) * (vb + vm) )
        # dpdx  = 0.5  * (pr - pl)
        dv2dy = 0.25 * ( (vt + vm) ** 2 - (vb + vm) ** 2 )
        dvudx = 0.25 * ( (vr + vm) * (ur + um) - (vl + vm) * (ul + um) )
        # dpdy  = 0.5  * (pt - pb)
        # d(Txx)/dx + d(Tyx)/dy = 1/Re * [du/d(x^2) + du/d(y^2)]
        # d(Txy)/dx + d(Tyy)/dy = 1/Re * [dv/d(x^2) + dv/d(y^2)]
        dudx2 = (ur - um * 2 + ul)
        dudy2 = (ut - um * 2 + ub)
        dvdx2 = (vr - vm * 2 + vl)
        dvdy2 = (vt - vm * 2 + vb)

        du_2dx  = 0.25 * ( (ur_ + um_) ** 2 - (ul_ + um_) ** 2 )
        du_v_dy = 0.25 * ( (ut_ + um_) * (vt_ + vm_) - (ub_ + um_) * (vb_ + vm_) )
        # dp_dx   = 0.5  * (pr_ - pl_)
        dv_2dy  = 0.25 * ( (vt_ + vm_) ** 2 - (vb_ + vm_) ** 2 )
        dv_u_dx = 0.25 * ( (vr_ + vm_) * (ur_ + um_) - (vl_ + vm_) * (ul_ + um_) )
        # dp_dy   = 0.5  * (pt_ - pb_)
        du_dx2  = (ur_ - um_ * 2 + ul_)
        du_dy2  = (ut_ - um_ * 2 + ub_)
        dv_dx2  = (vr_ - vm_ * 2 + vl_)
        dv_dy2  = (vt_ - vm_ * 2 + vb_)

        # TODO or use abs for each item of the conservation law
        lossMom0 = torch.abs((du2dx  +duvdy  -(dudx2 +dudy2) *iRe) -  \
                             (du_2dx +du_v_dy-(du_dx2+du_dy2)*iRe)) + \
                   torch.abs((dvudx  +dv2dy   -(dvdx2 +dvdy2) *iRe) -  \
                             (dv_u_dx+dv_2dy -(dv_dx2+dv_dy2)*iRe))
        lossMom = torch.mean(lossMom0)

        
        # dudx,dvdy,dpdx,dpdy have been computed
        dudy  = (ut  - ub ) * 0.5
        dvdx  = (vr  - vl ) * 0.5
        du_dy = (ut_ - ub_) * 0.5
        dv_dx = (vr_ - vl_) * 0.5
        # lossGdl is similar to lossMass
        lossGdl0 = torch.abs(dudx-du_dx) + torch.abs(dudy-du_dy) + \
                   torch.abs(dvdx-dv_dx) + torch.abs(dvdy-dv_dy) 
                   #+ \
                   #torch.abs(dpdx-dp_dx) + torch.abs(dpdy-dp_dy)
        lossGdl = torch.mean(lossGdl0) / 4.
        

        lossL10 = torch.abs(pred - trut)
        lossL1 = torch.mean(lossL10)
		
        loss = (lossMass*5 + lossMom*25 + lossL1) / 3
        return loss
