! ============================================================================
! Name        : FortranClassTest.f90
! Author      : Ioannis Stefanou
! Version     : 1.0
! Copyright   : All rights reserved 2018
! Description : Core plasticity code in Fortran after refactoring
! ============================================================================

!subroutines for general matrix operations
module matrix_operations
    implicit none
    contains

    !define scaled vector
    subroutine scale_vector(x,d,n,sx)
        real(8), intent(in) :: x(n),d(n)
        real(8), intent(out) :: sx(n)
        integer,intent(in) :: n
        integer i
        do i=1,n
            sx(i)=x(i)*d(i)
        end do
    end subroutine

    !define scaled vector infinite norm
    subroutine getscalednorm_inf(vector,d,m,rmax)
        real(8), intent(in) :: vector(m),d(m)
        real(8), intent(out) :: rmax
        integer, intent(in) :: m
        real(8) el
        integer i
        rmax=0.d0
        do i=1,m
            el=abs(d(i)*vector(i))
            if (el.gt.rmax) then
                rmax=el
            end if
        enddo
    end subroutine

    !define scaled vector norm
    subroutine getscalednorm(vector,d,m,rnorm)
        real(8), intent(in) :: vector(m),d(m)
        real(8), intent(out) :: rnorm
        integer, intent(in) :: m
        integer i
        rnorm=0.d0
        do i=1,m
            rnorm=rnorm+(d(i)*vector(i))**2
        enddo
        rnorm=dsqrt(rnorm)
    end subroutine

    !define vector norm
    subroutine getnorm(vector,m,rnorm)
        real(8), intent(in) :: vector(m)
        real(8), intent(out) :: rnorm
        integer, intent(in) :: m
        integer i
        rnorm=0.d0
        do i=1,m
            rnorm=rnorm+vector(i)**2
        enddo
        rnorm=dsqrt(rnorm)
    end subroutine

    !define vector square norm: 1/2*x_i*x_i
    subroutine getnorm2(vector,m,rnorm)
        real(8), intent(in) :: vector(m)
        real(8), intent(out) :: rnorm
        integer, intent(in) :: m
        integer i
        rnorm=0.d0
        do i=1,m
            rnorm=rnorm+vector(i)**2
        enddo
        rnorm=rnorm/2.d0
    end subroutine

    !get vector elements sign
    subroutine checksign(vector,m,rsign,rtol)
        real(8), intent(in) :: vector(m),rtol
        real(8), intent(out) :: rsign
        integer, intent(in) :: m
        integer i
        real(8) rmtol
        rmtol=-rtol
        rsign=+1.d0
        do i=1,m
            if (vector(i).lt.rmtol) then
                rsign=-1.d0
                return
            end if
        enddo
    end subroutine

    !form identity matrix
    subroutine getid(rid,n)
        real(8), intent(out) :: rid(n,n)
        integer, intent(in) :: n
        integer k1
        rid=0.d0
        do k1=1,n
            rid(k1,k1)=1.d0
        end do
    end subroutine

    !inverse order of vector elements
    subroutine invnorder(norder,ninvorder,nf)
        implicit none
        integer, intent(in) :: norder(nf)
        integer, intent(out) :: ninvorder(nf)
        integer, intent(in) :: nf
        integer i
        do i=1,nf
        ninvorder(norder(i))=i
        end do
    end subroutine

    ! calculate the pseudo inverse matrix
    subroutine pseudoinversela(a,ra,m,n,nill)

        !      interface
        !        subroutine dgesvd(r, area)
        !            real, intent(in) :: r
        !            real, intent(out) :: area
        !        end subroutine compute_area
        !      end interface

        real(8), intent(in) :: a(m,n)
        integer, intent(in) :: m,n
        integer, intent(out) :: nill
        real(8), intent(out) :: ra(m,n)

        integer, parameter :: lwmax=1000
        integer          info,lwork,i

        real(8) u(m,m),vt(n,n),s(n),work(lwmax)
        real(8) sp(n,n),rpa(m,n)
        real(8), parameter :: rsvdtol=1.d-10
        !.. external subroutines ..
        external dgesvd
        !.. intrinsic functions ..
        intrinsic int, min
        nill=0
        rpa=a
        ra=0.d0

        !query the optimal workspace.
        lwork = -1
        call dgesvd( 'all', 'all', m, n, rpa, m, s, u, m, vt, n, work, lwork, info )
        lwork = min( lwmax, int( work( 1 ) ) )
        !compute svd.
        call dgesvd( 'all', 'all', m, n, rpa, m, s, u, m, vt, n, work, lwork, info )
        !check for convergence.
        if( info.gt.0 ) then
         !write(*,*)'the algorithm computing svd failed to converge.'
            nill=1
            return
        end if

        if (m.ne.n) then
            return
        end if

        sp=0.d0
        do i=1,n
            if (dabs(s(i)).gt.rsvdtol) then
                sp(i,i)=s(i)**(-1)
            end if
        end do

        ra=matmul(matmul(transpose(vt),sp),transpose(u))

    end subroutine

end module


module plasticity_model
    implicit none

    type state
        integer nstr,nf,nh,nsvars,na,nill,na_prev
        real(8),allocatable :: stress(:)
        real(8),allocatable :: de_tot(:),de_pl(:),de_el(:),e_el(:),e_el0(:)
        real(8),allocatable :: del(:,:),ridel(:,:)
!        real(8),allocatable :: rf(:),rlamda(:),rq(:),rq0(:)
!        real(8),allocatable :: rfs(:,:),rgs(:,:),rhq(:,:),rfq(:,:)
!        real(8),allocatable :: rfss(:,:,:),rgss(:,:,:),rgqs(:,:,:),rhqs(:,:,:),rhqq(:,:,:)
        real(8),allocatable :: jac(:,:),rb(:)
        real(8),allocatable :: invjac(:,:)
        integer,allocatable :: norder(:),ninvorder(:)
        ! sorted with activated surfaces
        real(8),allocatable :: raf(:),ralamda(:),raq(:),raq0(:)
        real(8),allocatable :: rafs(:,:),rags(:,:),rahq(:,:),rafq(:,:)
        real(8),allocatable :: rafss(:,:,:),ragss(:,:,:),ragqs(:,:,:),rahqs(:,:,:),rahqq(:,:,:)
        ! properties
        integer :: nprops
        real(8) :: rtol,dtime
        real(8),allocatable :: mat_props(:),rvisc(:),sc_fsurf(:),sc_stresses(:),sc_lambdas(:),sc_hardening(:)
        !sorted
        real(8),allocatable :: ravisc(:),sc_afsurf(:),sc_alambdas(:),sc_ahardening(:)
      contains
        procedure :: update_plasticity => update_plasticity
        procedure :: set_jac_res => get_jac_res
        procedure :: error => get_error
        procedure :: get_mep => get_mep
        procedure :: refine_surf => refine_surf
        procedure :: check_surf => check_surf
        procedure :: check_lamda_sign => check_lamda_sign
    end type
    interface state
        module procedure init_state
    end interface

  contains
    ! initialize state
    function init_state(nstr,nf,nh,nsvars,svars,stress,de,props,nprops,rtol,dtime)
        use matrix_operations
        real(8), intent(in) :: svars(nsvars),stress(nstr),de(nstr),props(nprops),rtol,dtime
        integer, intent(in) :: nstr,nf,nh,nsvars,nprops
        integer i
        type(state) init_state
        init_state%dtime=dtime
        init_state%rtol=rtol
        init_state%nstr=nstr
        init_state%nf=nf
        init_state%na=0
        init_state%na_prev=0
        init_state%nh=nh
        init_state%nsvars=nsvars
        allocate(init_state%stress(nstr))
        allocate(init_state%de_tot(nstr),init_state%de_pl(nstr),init_state%de_el(nstr),init_state%e_el(nstr),init_state%e_el0(nstr))
        allocate(init_state%del(nstr,nstr),init_state%ridel(nstr,nstr))
!        allocate(init_state%rf(nf),init_state%rlamda(nf))
!        allocate(init_state%rfs(nstr,nf),init_state%rgs(nstr,nf))
!        allocate(init_state%rfss(nstr,nstr,nf),init_state%rgss(nstr,nstr,nf))
!        allocate(init_state%rq(nh),init_state%rq0(nh),init_state%rhq(nh,nf),init_state%rfq(nh,nf))
!        allocate(init_state%rgqs(nh,nstr,nf),init_state%rhqs(nh,nstr,nf),init_state%rhqq(nh,nh,nf))
        allocate(init_state%norder(nf),init_state%ninvorder(nf))
        ! sorted
        allocate(init_state%raf(nf),init_state%ralamda(nf))
        allocate(init_state%rafs(nstr,nf),init_state%rags(nstr,nf))
        allocate(init_state%rafss(nstr,nstr,nf),init_state%ragss(nstr,nstr,nf))
        allocate(init_state%raq(nh),init_state%raq0(nh),init_state%rahq(nh,nf),init_state%rafq(nh,nf))
        allocate(init_state%ragqs(nh,nstr,nf),init_state%rahqs(nh,nstr,nf),init_state%rahqq(nh,nh,nf))

        if (nh.gt.0) then
            init_state%raq=svars(2*nstr+1:2*nstr+nh)
            init_state%raq0=svars(2*nstr+1:2*nstr+nh)
        end if
        init_state%ralamda=0.d0
        forall (i=1:nf) init_state%norder(i)=i; init_state%ninvorder=init_state%norder
        init_state%stress=stress
        call getelstrain(init_state%e_el0,stress,nstr,init_state%raq,nh,props,nprops)
        init_state%e_el=init_state%e_el0+de

        call getelstress(init_state%stress,init_state%e_el,nstr,init_state%raq,nh,props,nprops)
        call getelmatrix(init_state%del,init_state%stress,nstr,init_state%raq,nh,props,nprops)
        call pseudoinversela(init_state%del,init_state%ridel,nstr,nstr,init_state%nill);
        if (init_state%nill.eq.1) write(6,*) "Unexpected error at inversion of elasticity matrix";
        init_state%de_pl=0.d0
        init_state%de_el=de
        init_state%de_tot=de
        ! set material properties and algorithm parameters
        allocate(init_state%mat_props(nprops),init_state%rvisc(nf))
        allocate(init_state%sc_fsurf(nf),init_state%sc_stresses(nstr),init_state%sc_lambdas(nf), &
            init_state%sc_hardening(nh))
        !sorted
        allocate(init_state%ravisc(nf),init_state%sc_afsurf(nf),init_state%sc_alambdas(nf), &
            init_state%sc_ahardening(nh))
        init_state%mat_props=props
        init_state%nprops=nprops
        call getviscparams(init_state%ravisc,nf,props,nprops)
        call getscaling(init_state%sc_fsurf,nf,init_state%sc_stresses,nstr, &
                init_state%sc_lambdas,nf, init_state%sc_hardening,nh)
        ! allocate jacobian and residual
        allocate(init_state%jac(nf+nstr+nh,nf+nstr+nh),init_state%rb(nf+nstr+nh))
        allocate(init_state%invjac(nf+nstr+nh,nf+nstr+nh))
    end function

    subroutine update_plasticity(me,rdx)
        use matrix_operations
        class(state), intent(inout) :: me
        real(8), intent(in) :: rdx(me%nf+me%nstr+me%nh)
        me%stress=me%stress+rdx(1:me%nstr)
        me%ralamda(1:me%na)=me%ralamda(1:me%na)+rdx(me%nstr+1:me%nstr+me%na)
        if (me%nh.gt.0) me%raq(1:me%nh)=me%raq(1:me%nh)+rdx(me%na+me%nstr+1:me%na+me%nstr+me%nh)

!        call calcsurf(me%rf,me%nf,me%stress,me%nstr,me%rq,me%nh,me%mat_props,me%nprops,me%norder)
!        call calcgradsurf(me%rfs,me%rgs,me%nf,me%stress,me%nstr,me%rq,me%nh,me%mat_props,me%nprops,me%norder)
!        call calcsecgradsurf(me%rfss,me%rgss,me%nf,me%stress,me%nstr,me%rq,me%nh,me%mat_props,me%nprops,me%norder)
!        if (me%nh.ne.0) call calchardparams(me%rhq,me%rfq,me%rhqq,me%rhqs,me%rgqs,me%rq,me%stress, &
!                                me%nh,me%nstr,me%nf,me%mat_props,me%nprops,me%norder)

        call calcsurf(me%raf,me%nf,me%stress,me%nstr,me%raq,me%nh,me%mat_props,me%nprops,me%norder)
        call calcgradsurf(me%rafs,me%rags,me%nf,me%stress,me%nstr,me%raq,me%nh,me%mat_props,me%nprops,me%norder)
        call calcsecgradsurf(me%rafss,me%ragss,me%nf,me%stress,me%nstr,me%raq,me%nh,me%mat_props,me%nprops,me%norder)
        if (me%nh.ne.0) call calchardparams(me%rahq,me%rafq,me%rahqq,me%rahqs,me%ragqs,me%raq,me%stress, &
                                me%nh,me%nstr,me%nf,me%mat_props,me%nprops,me%norder)

        call getelstrain(me%e_el,me%stress,me%nstr,me%raq,me%nh,me%mat_props,me%nprops)
        call getelmatrix(me%del,me%stress,me%nstr,me%raq,me%nh,me%mat_props,me%nprops)
        call pseudoinversela(me%del,me%ridel,me%nstr,me%nstr,me%nill);
        if (me%nill.eq.1) write(6,*) "Unexpected error at inversion of elasticity matrix";
        me%de_pl=matmul(me%rags(1:me%nstr,1:me%na),me%ralamda(1:me%na))
    end subroutine

    subroutine refine_surf(me)
        use matrix_operations
        class(state), intent(inout) :: me
        integer i,na,j,nna
        integer nmrspm(me%nf)
        real(8) traf(me%nf),tralamda(me%nf), &
            trafs(me%nstr,me%nf),trags(me%nstr,me%nf), &
            trafss(me%nstr,me%nstr,me%nf),tragss(me%nstr,me%nstr,me%nf), &
            traq(me%nh),trahq(me%nh,me%nf),trafq(me%nh,me%nf), &
            tragqs(me%nh,me%nstr,me%nf),trahqs(me%nh,me%nstr,me%nf),trahqq(me%nh,me%nh,me%nf), &
            travisc(me%nf),tsc_afsurf(me%nf),tsc_alambdas(me%nf)
        me%na_prev=me%na
        na=0
        nna=me%nf+1
        do i=1,me%nf
        j=me%norder(i)
        if (me%raf(j).ge.me%rtol) then
            na=na+1
            traf(na)=me%raf(j)
            trafs(:,na)=me%rafs(:,j)
            trags(:,na)=me%rags(:,j)
            trafss(:,:,na)=me%rafss(:,:,j)
            tragss(:,:,na)=me%ragss(:,:,j)
            !trahq(:,na)=me%rahq(:,j)
            trafq(:,na)=me%rafq(:,j)
            !trahqs(:,:,na)=me%rahqs(:,:,j)
            !trahqq(:,:,na)=me%rahqq(:,:,j)
            tragqs(:,:,na)=me%ragqs(:,:,j)
            tralamda(na)=me%ralamda(j)
            travisc(na)=me%ravisc(j)
            tsc_afsurf(na)=me%sc_afsurf(j)
            tsc_alambdas(na)=me%sc_alambdas(j)
            nmrspm(j)=na
        else
            nna=nna-1
            traf(nna)=me%raf(j)
            trafs(:,nna)=me%rafs(:,j)
            trags(:,nna)=me%rags(:,j)
            trafss(:,:,nna)=me%rafss(:,:,j)
            tragss(:,:,nna)=me%ragss(:,:,j)
            !trahq(:,nna)=me%rahq(:,j)
            trafq(:,nna)=me%rafq(:,j)
            !trahqs(:,:,nna)=me%rahqs(:,:,j)
            !trahqq(:,:,nna)=me%rahqq(:,:,j)
            tragqs(:,:,nna)=me%ragqs(:,:,j)
            tralamda(nna)=me%ralamda(j)
            travisc(nna)=me%ravisc(j)
            tsc_afsurf(nna)=me%sc_afsurf(j)
            tsc_alambdas(nna)=me%sc_alambdas(j)
            nmrspm(j)=nna
        end if
      end do

      me%raf=traf;me%rafs=trafs;me%rags=trags;me%rafss=trafss;me%ragss=tragss
      me%rafq=trafq;me%ragqs=tragqs
      me%ralamda=tralamda;me%ravisc=travisc;me%sc_afsurf=tsc_afsurf;me%sc_alambdas=tsc_alambdas
      !me%rahq=trahq;me%rahqs=trahqs;me%rahqq=trahqq;
      me%na=na
      me%norder=nmrspm
      call invnorder(me%norder,me%ninvorder,me%nf)
    end subroutine

    function get_error(me,rdx)
        use matrix_operations
        class(state), intent(inout) :: me
        real(8), intent(in) :: rdx(me%nf+me%nstr+me%nh)
        real(8),allocatable :: scalevector(:)
        real(8) :: get_error
        integer n
        real(8) error
        n=me%na+me%nstr+me%nh
        !stress change
        allocate(scalevector(n))
        scalevector=1.d0
        scalevector(1:me%nstr)=me%sc_stresses
        !has to change in case of multisurface plasticity -> look previous versions (seen as minimization problem)
        !call getscalednorm(rdx,scalevector,n,error)
        call getnorm(rdx,n,error)
        get_error=error
        scalevector=1.d0
        !lamda change
        scalevector(me%nstr+1:me%nstr+me%na)=me%sc_afsurf(1:me%na)
        !call getscalednorm(me%rb,scalevector,n,error)
        call getnorm(me%rb,n,error)
        get_error=max(get_error,error)
    end function

    function check_surf(me)
        class(state), intent(in) :: me
        integer :: check_surf
        integer i
        check_surf=1
        do i=1,me%nf
            if (me%raf(i).ge.me%rtol) return
        end do
        check_surf=0
    end function

    function check_lamda_sign(me)
        class(state), intent(in) :: me
        integer :: check_lamda_sign
        integer i
        check_lamda_sign=1
        do i=1,me%na
            if (me%ralamda(i).lt.-me%rtol) return
        end do
        check_lamda_sign=0
    end function

    subroutine get_jac_res(me)
        class(state), intent(inout) :: me
        integer na,nh,nstr,i
        real(8) tmp(me%nstr,me%nh),rihd(me%nh,me%nh)

        me%jac=0.d0
        ! calculate jacobian
        na=me%na;nstr=me%nstr;nh=me%nh
        forall (i=1:nstr) me%jac(i,1:nstr)=matmul(me%ragss(i,1:nstr,1:na),me%ralamda(1:na))

        me%jac(1:nstr,1:nstr)=me%ridel+me%jac(1:nstr,1:nstr)
        me%jac(1:nstr,nstr+1:nstr+na)=me%rags(:,1:na)
        me%jac(nstr+1:nstr+na,1:nstr)=transpose(me%rafs(:,1:na))
        !add viscoplastic terms
        if (me%dtime.ne.0.d0) then
            forall (i=1:na) me%jac(nstr+i,nstr+i)=-me%ravisc(i)/me%dtime
        end if
        ! calculate residual
        me%rb=0.d0
        me%rb(1:nstr)=me%de_tot-me%e_el+me%e_el0
        me%rb(1:nstr)=me%rb(1:nstr)-matmul(me%rags(:,1:na),me%ralamda(1:na))
        ! contribution of failure surface to residual
        me%rb(nstr+1:nstr+na)=-me%raf(1:na)
        ! contribution of viscoplasticity to residual
        if (me%dtime.ne.0.d0) then
            forall (i=1:na) me%rb(nstr+i)=me%rb(nstr+i)+me%ravisc(i)*me%ralamda(i)/me%dtime
        end if

        if (nh.eq.0) return

        ! hardening contribution to residual and jacobian
        !first derivatives
        !lines 1 - nstr
        call getdeeldb(tmp,me%stress,nstr,me%raq,nh,me%mat_props,me%nprops)
        me%jac(1:nstr,nstr+na+1:nstr+na+nh)=tmp
        !lines nstr+1 - nstr+na
        me%jac(nstr+1:nstr+na,nstr+na+1:nstr+na+nh)=transpose(me%rafq(:,1:na))
        !lines nstr+na+1 - nstr+na+nh
        me%jac(na+nstr+1:na+nstr+nh,nstr+1:nstr+na)=me%rahq(:,1:na)
        !second derivatives
        !lines 1 - nstr
        forall (i=1:nstr) me%jac(i,na+nstr+1:na+nstr+nh)= &
              me%jac(i,na+nstr+1:na+nstr+nh)+matmul(me%ragqs(1:nh,i,1:na),me%ralamda(1:na)) !changed to ralamda -> check
        !lines nstr+na+1 - nstr+na+nh
        do i=1,nh
            me%jac(nstr+na+i,1:nstr)= &
                    me%jac(nstr+na+i,1:nstr)+matmul(me%rahqs(i,1:nstr,1:na),me%ralamda(1:na))
            me%jac(nstr+na+i,nstr+na+1:nstr+na+nh)= &
                    me%jac(nstr+na+i,nstr+na+1:nstr+na+nh)+matmul(me%rahqq(i,1:nh,1:na),me%ralamda(1:na)) !changed to ralamda -> check
        end do
        rihd=0.d0; forall (i=1:nh) rihd(i,i)=1.d0
        me%jac(nstr+na+1:nstr+na+nh,nstr+na+1:nstr+na+nh)=-rihd+ &
            me%jac(nstr+na+1:nstr+na+nh,nstr+na+1:nstr+na+nh)
        me%rb(nstr+na+1:nstr+na+nh)=me%raq(1:nh)- &
            me%raq0(1:nh)-matmul(me%rahq(1:nh,1:na),me%ralamda(1:na)) !changed to ralamda and to raq and raq0-> check
        !print*,"rjac",me%jac
        !print*,"rb",me%rb
    end subroutine

subroutine get_mep_old(me,dsde)
        use matrix_operations
        implicit none
        class(state), intent(inout) :: me
        real(8), intent(out) :: dsde(me%nstr,me%nstr)
        real(8) TH(me%nstr,me%nstr),invTH(me%nstr,me%nstr),Mmatrix(1:me%na,1:me%na),invMmatrix(1:me%na,1:me%na)
        real(8) psi(1:me%nh,1:me%nh),invpsi(1:me%nh,1:me%nh),rmtmp1(1:me%na,1:me%nstr)
        real(8) ratio
        integer nstr,na,nh,i,nill
        !calculate consistent dsde
        nstr=me%nstr;na=me%na;nh=me%nh
        !calculate consistent dsde
        !determine dg/dss * lamda_dot
        forall (i=1:nstr) invTH(i,1:nstr)=matmul(me%ragss(i,1:nstr,1:na),me%ralamda(1:na)) !changed to ralamda -> check
        !determine TH^-1 = C + dg/dss * lamda_dot
        invTH=me%ridel+invTH
        !inverse -> TH
        call pseudoinversela(invTH(1:nstr,1:nstr),TH(1:nstr,1:nstr),nstr,nstr,nill)
        if (nill.eq.1) write(6,*) "Unexpected error at inversion of TH matrix";
        !determine (df/ds)t * TH = rmtmp1(1:na,1:nstr)
        rmtmp1(1:na,1:nstr)= &
            matmul(transpose(me%rafs(:,1:na)),TH)
        !***do not replace RMTMP1 until ds/de is calculated***
        !determine (df/ds)T * TH * dg/ds = RMTMP3(1:NA,1:NA)
        Mmatrix(1:na,1:na)=matmul(rmtmp1(1:na,1:nstr),me%rags(1:nstr,1:na))
        !this is the M matrix (without the contrinution of hardening)
        !update M matrix for hardening before inversing
        if (nh.ne.0) then
            !determine dhq/dqq*lamda_dot
!            forall(i=1:nh) invpsi(i,1:nh)=-matmul(me%rahqq(i,1:nh,1:na),me%ralamda(1:na)) !changed to ralamda -> check
            !determine PSI^-1 = I - dhq/dqq*lamda_dot
            forall(i=1:nh) invpsi(i,i)=1.d0+invpsi(i,i)
            !determine PSI
            call pseudoinversela(invpsi,psi,nh,nh,nill)
            if (nill.eq.1) write(6,*) "Unexpected error at inversion of PSI matrix";
            !determine M = (df/ds)T * TH * dg/ds - (df/dq)T * PSI * hq
            ! initially with -
            Mmatrix = Mmatrix + matmul(matmul(transpose(me%rafq(1:nh,1:na)),psi),me%rahq(1:nh,1:na))
        end if
        call pseudoinversela(mMatrix,invMmatrix,na,na,nill)
        if (nill.eq.1) write(6,*) "Unexpected error at inversion of M matrix";
        !determine I  - dg/ds * M^-1 * (df/ds)T * TH
        dsde=-matmul(me%rags(1:nstr,1:na),matmul(invMmatrix,rmtmp1))
        forall (i=1:nstr) dsde(i,i)=1.d0+dsde(i,i)
        !determine TH * (I  - dg/ds * M^-1 * (df/ds)T * TH)
        dsde=matmul(TH,dsde)
        !this is the ds/de
        ratio=0.d0
        dsde=ratio*me%del+(1.d0-ratio)*dsde !add a small part of elasticity

    end subroutine


    subroutine get_mep(me,dsde)
        use matrix_operations
        implicit none
        class(state), intent(inout) :: me
        real(8), intent(out) :: dsde(me%nstr,me%nstr)
        real(8) ratio
!        write(6,*)  'ALECX 28/04'
        !calculate consistent dsde
        dsde=me%invjac(1:me%nstr,1:me%nstr)
        !this is the ds/de
        !ratio=1.d-1
        !ratio=1.d0*1.d-1
        ratio=0.d0
        dsde=ratio*me%del+(1.d0-ratio)*dsde !add a small part of elasticity

    end subroutine

end module plasticity_model


subroutine usermatmel(stress,de,dsde,nstr,props,nprops,svarsgp,nsvarsgp,nill)
    use matrix_operations
    use plasticity_model
    implicit none
    real(8), intent(inout) :: stress(nstr),dsde(nstr,nstr),svarsgp(nsvarsgp),props(nprops)
    integer, intent(out) :: nill
    real(8), intent(in) :: de(nstr)
    integer, intent(in) :: nstr,nprops,nsvarsgp
    real(8) :: dtime=1.d0
    !real(8) :: rtol=1.d-6
    integer :: nf=0,nh=0

    call usermatmpl(stress,de,dsde,nstr,props,nprops,nf,nh,svarsgp,nsvarsgp,dtime,nill)

end subroutine

subroutine usermatmpl(stress,de,dsde,nstr,props,nprops,nf,nh,svarsgp,nsvarsgp,dtime,nill)
    use matrix_operations
    use plasticity_model
    use, intrinsic :: ieee_arithmetic
    implicit none
    real(8), intent(inout) :: stress(nstr),dsde(nstr,nstr),svarsgp(nsvarsgp),props(nprops)
    integer, intent(out) :: nill
    real(8), intent(in) :: de(nstr)
    integer, intent(in) :: nstr,nf,nh,nprops,nsvarsgp

    real(8) :: rtol=1.d-8!1.d-5
    integer :: maxiter=50, maxnsiter=50

    integer nsiter,iter,na
    type(state) :: state_tdt, state_t
    real(8) :: rdx(nf+nstr+nh),rcrit1,dtime

    nill=0; nsiter=0; rdx=0.d0
    !calculate trial stress/elasticity
    state_tdt=state(nstr,nf,nh,nsvarsgp,svarsgp,stress,de,props,nprops,rtol,dtime)
    if (nf.eq.0) then
        stress=state_tdt%stress
        dsde=state_tdt%del
        call update_svars(state_tdt,svarsgp,nsvarsgp)
        return
    end if

!    write(6,*) 'ALECX 28/04'
!    write(6,*) stress
!    write(6,*) de
!    write(6,*) dsde
!    write(6,*) nstr
!    write(6,*) props
!    write(6,*) nprops
!    write(6,*) nf
!    write(6,*) nh
!    write(6,*) svarsgp
!    write(6,*) nsvarsgp
!    write(6,*) dtime
!    write(6,*) nill
!    write(6,*) 'ALECX 28/04'

    ! calculate plastic material
    call state_tdt%update_plasticity(rdx)
    call state_tdt%refine_surf()
    ! if inside the elastic domain na=0 -> update and exit
    na=state_tdt%na
    if (na.eq.0) then
        stress=state_tdt%stress
        dsde=state_tdt%del
        call update_svars(state_tdt,svarsgp,nsvarsgp)
        return
    end if

    call state_tdt%set_jac_res()
    ! iterator
    iter=1
    do while (iter.ge.1)

        call linesearch(state_tdt,rdx(1:na+nstr+nh),0,rcrit1)
        rcrit1=state_tdt%error(rdx)

        iter=iter+1

        if (isnan(rcrit1)) then
            write(6,*) '>error: nan residual at material'
            nill=1
            return
        elseif (abs(rcrit1) >= huge(rcrit1)) then
            write(6,*) '>error: infinite residual at material'
            nill=1
            return
        end if
        if (rcrit1.le.rtol) then
            iter=0
        elseif ((iter.gt.maxiter).or.(state_tdt%nill.eq.1)) then
            nill=1
            return
!        else
!            write(6,*) 'unhandled',iter,rcrit1
        end if
    end do

    !check sign of lagrange multipliers
    if (state_tdt%check_lamda_sign().gt.0) then!.or.(state_tdt%check_surf().gt.0)) then
        !re-set active surfaces
        write(6,*) '>error: multisurface plasticity algorithm &
            negative lamda or outside elastic domain'
        print*, ">lamda:",state_tdt%ralamda
        print*, ">fsurf:",state_tdt%raf
        nill=1
        return
        ! has to do as in previous versions for multisurface plasticity
        call state_tdt%refine_surf()
        nsiter=nsiter+1
        !goto ...
        if (nsiter.ge.maxnsiter) then
          write(6,*) '>error: multisurface plasticity alogrithm &
            failed to convergence (in active surfaces updating) after iterations:',nsiter
            nill=1
            return
            ! has to do as in previous versions for multisurface plasticity
        end if

    end if
!    WRITE(6,*) 'CONVERGED'
    !CONVERGED
    stress=state_tdt%stress

    call state_tdt%get_mep(dsde)
    call update_svars(state_tdt,svarsgp,nsvarsgp)
!    WRITE(6,*) 'CONVERGED1'

end subroutine

subroutine linesearch(state_tdt,du,info,rr)
    use plasticity_model
    use matrix_operations
    implicit none
    type(state),intent(inout) :: state_tdt
    type(state) :: state_tmp
    real(8), intent(out), dimension(state_tdt%na+state_tdt%nstr+state_tdt%nh) :: du
    real(8) :: l0,l1,fc,f0,f1,alpha,ltemp,lmin,initslope
    real(8) :: rijac(state_tdt%na+state_tdt%nstr+state_tdt%nh,state_tdt%na+state_tdt%nstr+state_tdt%nh)
    real(8) :: rdx(state_tdt%na+state_tdt%nstr+state_tdt%nh),rcrit1
    real(8) :: m1(2,2),m2(2),rr
    integer n,na,nh,nstr,nill,iter,info

    nstr=state_tdt%nstr;na=state_tdt%na;nh=state_tdt%nh;n=na+nstr+nh

    call getnorm2(state_tdt%rb(1:n),n,fc)
    call pseudoinversela(state_tdt%jac(1:n,1:n), &
            rijac,n,n,nill)
    state_tdt%invjac(1:n,1:n)=rijac
    du=matmul(rijac,state_tdt%rb(1:n))
    rdx=du
    initslope=-2.d0*fc
    call state_tdt%update_plasticity(du)
    call state_tdt%set_jac_res()
    call getnorm2(state_tdt%rb(1:n),n,f1)
    l1=1.d0;f0=fc

    if ((f1 <= fc + alpha*l1*initslope).or.(f1 <= 5.d-1*state_tdt%rtol**2)) then
        if (info.eq.1) print*, "no line search"
        return
    end if
    ! entering linesearch algorithm
    call getnorm2(du,n,alpha)

    alpha=1.d-4
    lmin=state_tdt%rtol*alpha
    iter=1
    do while (iter.ge.1)
        if (f1 <= fc + alpha*l1*initslope) then
            if (info.eq.1) print*, "finished",l1,iter,fc,f1
            du=rdx
            state_tdt=state_tmp
            return
        elseif (l1.lt.lmin) then
            if (info.eq.1) print*,"lmin",l1
            rr=f1
            return !lamda too small, line search failed, using initial large du
        else !find new lamda
            state_tmp=state_tdt
            if (l1.eq.1.d0) then
                !print*, "quad approximation"
                !ltemp=f0/(f1+f0)
                ltemp=-initslope/(2.d0*(f1-fc-initslope))
            else
                !print*, "cubic approximation"
                m1(1,1)= 1.d0/l1**2
                m1(1,2)=-1.d0/l0**2
                m1(2,1)=-l0/l1**2
                m1(2,2)= l1/l0**2
                m2(1)= f1 - fc - l1*initslope
                m2(2)= f0 - fc - l0*initslope
                m2=(1.d0/(l1-l0))*matmul(m1,m2)
                if (m2(1).ne.0.d0) then
                    ltemp=(-m2(2)+Sqrt(m2(2)**2-3.d0*m2(1)*initslope))/(3.d0*m2(1))
                    !print*,"here1",ltemp,l1,l0,m2
                else
                    ltemp=-initslope/(2.d0*m2(2))
                    !print*,"here2",l1
                end if
                if (ltemp.gt.5.d-1*l0) ltemp=l0*5.d-1
            end if
            l0=l1
            f0=f1
            if (ltemp.lt.1.d-1*l1) then
                l1=l1*1.d-1
                !print*,"here3",l1
            else
                l1=ltemp
                !print*,"here4",l1
            endif
            rdx=l1*du
            call state_tmp%update_plasticity(rdx)
            call state_tmp%set_jac_res()
            call getnorm2(state_tmp%rb(1:n),n,f1)

        end if

        iter=iter+1
    end do



    !print*,"rdx",rdx

    ! backup and update
    !state_t=state_tdt

    call state_tdt%update_plasticity(rdx)
    call state_tdt%set_jac_res()

end subroutine

subroutine update_svars(state_tdt,svars,nsvars)
    use plasticity_model
    implicit none
    real(8), intent(inout) :: svars(nsvars)
    integer, intent(in) :: nsvars
    type(state),intent(in) :: state_tdt
    integer nh,nstr,nf
    nstr=state_tdt%nstr;nf=state_tdt%nf;nh=state_tdt%nh;
    svars(1:nstr)=state_tdt%stress
    svars(nstr+1:2*nstr)=svars(nstr+1:2*nstr)+state_tdt%de_tot
    nh=state_tdt%nh;nstr=state_tdt%nstr

    if (nh.ne.0) then
        svars(2*nstr+1:2*nstr+nh)=state_tdt%raq
    end if
    if (nsvars.ge.(3*nstr+nh)) then
        svars(2*nstr+1+nh:3*nstr+nh)=svars(2*nstr+1+nh:3*nstr+nh)+state_tdt%de_pl
    end if
    if (nsvars.ge.(3*nstr+1+nh)) then
        svars(3*nstr+1+nh:3*nstr+1+nh)=state_tdt%na
    end if
    if (nsvars.ge.(3*nstr+nf+1+nh)) then
        !svars(3*nstr+1+1+nh:3*nstr+nf+1+nh)=reorder(state_tdt%raf,state_tdt%ninvorder,nf)
    end if
    if (nsvars.ge.(3*nstr+2*nf+1+nh)) then
        svars(3*nstr+nf+1+1+nh:3*nstr+2*nf+1+nh)=reorder(state_tdt%ralamda,state_tdt%ninvorder,nf)
    end if
    if (nsvars.ge.(4*nstr+2*nf+1+nh)) then
        svars(3*nstr+2*nf+1+1+nh:4*nstr+2*nf+1+nh)=state_tdt%de_pl !state_tdt%e_el
    end if

  contains
    function reorder(vector,neworder,n)
        real(8), intent(in) :: vector(n)
        integer, intent(in) :: neworder(n),n
        real(8), dimension(n) :: reorder,tmp
        integer i
        forall (i=1:n) tmp(neworder(i))=vector(i)
        reorder=tmp
    end function
end subroutine



!        intent(in) means that the variable value can enter, but not be changed
!        intent(out) means the variable is set inside the procedure and sent back to the main program with any initial values ignored.
!        intent(inout) means that the variable comes in with a value and leaves with a value (default).

!real(8), optional, intent(inout) :: rtol
        !if( .not. present(rtol) ) rtol = dble(1.d-6)













