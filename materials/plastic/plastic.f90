module plastic

  integer, parameter :: rk=selected_real_kind(14)

  ! parameter pointers
  integer, parameter :: nui=5
  integer, parameter :: ipk=1
  integer, parameter :: ipmu=2

  ! yield surface parameters
  integer, parameter :: ipa1=3
  integer, parameter :: ipa4=4
  integer, parameter :: ipdev=5

  ! state variable pointers
  integer, parameter :: nxtra=2
  integer, parameter :: keqps=1
  integer, parameter :: keqpv=2

  ! numbers
  real(kind=rk), parameter :: half=.5_rk
  real(kind=rk), parameter :: zero=0._rk, one=1._rk, two=2._rk
  real(kind=rk), parameter :: three=3._rk, six=6._rk, ten=10._rk
  real(kind=rk), parameter :: refeps=.01_rk
  real(kind=rk), parameter :: tol1=1.e-8_rk, tol2=1.e-6_rk
  real(kind=rk), parameter :: toor3=5.7735026918962584E-01
  real(kind=rk), parameter :: root2=1.4142135623730951E+00
  real(kind=rk), parameter :: root3=1.7320508075688772E+00

  ! tensors
  real(kind=rk), parameter :: delta(6) = (/one, one, one, zero, zero, zero/)
  real(kind=rk), parameter :: ez(6) = (/toor3, toor3, toor3, zero, zero, zero/)

contains

  subroutine pchk(ui)
    ! ----------------------------------------------------------------------- !
    ! Check the validity of user inputs and set defaults.
    !
    ! In/Out
    ! ------
    ! ui : array
    !   User input
    !
    ! ----------------------------------------------------------------------- !
    implicit none
    !..................................................................parameters
    !......................................................................passed
    real(kind=rk), intent(inout) :: ui(*)
    !.......................................................................local
    character*12 iam
    parameter(iam='plastic_chk' )
    real(kind=rk) :: nu
    !~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ plastic_chk
    ! pass parameters to local variables
    ! check elastic moduli, calculate defaults if necessary
    if(ui(ipk) <= zero) then
       call faterr(iam, "Bulk modulus K must be positive")
    end if
    if(ui(ipmu) <= zero) then
       call faterr(iam, "Shear modulus MU must be positive")
    end if
    nu = (three * ui(ipk) - two * ui(ipmu)) / (six * ui(ipk) + two * ui(ipmu))
    if (nu > half) call faterr(iam, "Poisson's ratio > .5")
    if (nu < -one) call faterr(iam, "Poisson's ratio < -1.")
    if(nu < zero) call logmes("#---- WARNING: negative Poisson's ratio")

    ! check strength parameters
    if(ui(ipa1) <= zero) call faterr(iam, "Yield strength A1 must be positive")
    if(ui(ipa4) < zero) call faterr(iam, "A4 must be non-negative")

    return
  end subroutine pchk

  !*****************************************************************************!

  subroutine prxv(ui, nx, namea, keya, rinit)
    ! ----------------------------------------------------------------------- !
    ! Request extra variables and set defaults
    !
    ! In
    ! --
    ! ui : array
    !   User input
    !
    ! Out
    ! ---
    ! nx : int
    !   Number of extra variables
    ! namea : array
    !   Array of extra variable names
    ! keya : array
    !   Array of extra variable keys
    ! rinit : array
    !   Extra variable initial values
    ! ----------------------------------------------------------------------- !
    implicit none
    !..................................................................parameters
    integer, parameter :: mmcn=50, mmck=10
    integer, parameter :: mmcna=nxtra*mmcn, mmcka=nxtra*mmck
    !......................................................................passed
    real(kind=rk), intent(in) :: ui(nui)
    integer, intent(out) :: nx
    real(kind=rk), intent(out) :: rinit(nxtra)
    character(len=1), intent(out) :: namea(nxtra)
    character(len=1), intent(out) :: keya(nxtra)
    !.......................................................................local
    character(len=mmcn) :: name(nxtra)
    character(len=mmck) :: key(nxtra)
    real(kind=rk) :: dum
    !~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ plastic_rxv

    if (ui(ipdev) > zero) then
       call logmes('#---- requesting plastic extra variables')
    end if
    dum = ui(1)
    rinit(1:nxtra) = zero

    nx = 0

    ! equivalent plastic shear strain
    nx = nx + 1
    if(nx /= keqps) call bombed("keqps pointer wrong")
    name(nx) = "equivalent plastic shear strain"
    key(nx) = "EQPS"

    ! equivalent plastic volume strain
    nx = nx + 1
    if(nx /= keqpv) call bombed("keqpv pointer wrong")
    name(nx) = "equivalent plastic volume strain"
    key(nx) = "EQPV"

    call tokens(nx, name, namea)
    call tokens(nx, key, keya)
    return
  end subroutine prxv

  ! ************************************************************************* !

  subroutine pcalc(nx, dt, ui, d, stress, xtra)
    ! ----------------------------------------------------------------------- !
    ! Pressure dependent plasticity
    !
    ! In
    ! --
    ! nx : int
    !   Number of internal state vars
    ! dt : float
    !   Current time increment
    ! ui : array
    !   User inputs
    ! d : array
    !   Deformation rate
    !
    ! In/Out
    ! ------
    ! stress : array
    !   Stress
    ! xtra : array
    !   Material variables
    ! ----------------------------------------------------------------------- !
    implicit none
    ! ...................................................................passed
    integer, intent(in) :: nx
    real(kind=rk), intent(in) :: dt
    real(kind=rk), intent(in) :: ui(nui), d(6)
    real(kind=rk), intent(inout) :: stress(6), xtra(nx)
    ! ....................................................................local
    integer, parameter :: nit=10
    integer :: i, conv
    real(kind=rk) :: k, mu, c1, c4, de(6)
    real(kind=rk) :: zn, z, rn, r, f(nit), b(nit)
    real(kind=rk) :: t(6), er(6), dstress(6), dep(6)
    real(kind=rk) :: mz, mr, az, ar, gz, gr, refj2
    ! ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ plastic_calc

    ! Initialize local variables
    k = ui(ipk); mu = ui(ipmu)
    c1 = ui(ipa1) * root2
    c4 = ui(ipa4) * root2 * root3
    refj2 = two * mu * refeps
    b = zero; f = zero

    ! strain increment
    de = d * dt

    ! Rendulic components of initial stress
    call rendulic(stress, zn, rn)

    ! Trial stress state
    t = stress + three * k * iso(de) + two * mu * dev(de)
    call rendulic(t, z, r, er)

    ! evaluate yield function at trial state
    f(1) = yield(c1, c4, z, r)

    if (f(1) > zero) then

       conv = 0

       call grad(c1, c4, z, r, gz, gr)
       ! Rendulic components of flow direction
       mz = gz
       mr = gr
       ! Components of projection direction a = C:m
       az = three * k * mz
       ar = two * mu * mr

       b(2) = mag(de)
       f(2) = yield(c1, c4, z - b(2) * az, r - b(2) * ar)

       secant: do i = 3, nit

          ! secant step
          b(i) = b(i-1) - f(i-1) * (b(i-1) - b(i-2)) / (f(i-1) - f(i-2))

          ! yield function gradient
          call grad(c1, c4, z - b(i) * az, r - b(i) * ar, gz, gr)
          mz = gz; mr = gr
          az = three * k * mz; ar = two * mu * mr
          f(i) = yield(c1, c4, z - b(i) * az, r - b(i) * ar)

          if(abs(f(i)) / refj2 < tol1) then
             conv = 1
             z = z - b(i) * az
             r = r - b(i) * ar
             exit secant
          end if

       end do secant

       if (conv == 0) then
          call bombed("Secant iterations failed to converge")
       end if

       if (r <= zero) then
          ! Put stress on vertex with hydrosat
          if (c4 == zero) then
             call bombed("r < 0")
          end if
          r = zero
          z = c1 / c4
       end if

       ! Compute updated stress
       t = z * ez + r * er

       ! Compute plastic strain
       dstress = t - stress
       dep = de - one / three / k * iso(dstress) + one / two / mu * dev(dstress)

       xtra(keqps) = xtra(keqps) + mag(dev(dep))
       xtra(keqpv) = xtra(keqpv) + trace(dep)

    end if

    stress = t

    return

  end subroutine pcalc

  !***************************************************************************!

  function yield(c1, c4, z, r)
    ! ----------------------------------------------------------------------- !
    ! Evaluate yield function
    ! ----------------------------------------------------------------------- !
    implicit none
    real(kind=rk), intent(in) :: c1, c4, z, r
    real(kind=rk) :: yield
    yield = r - (c1 - c4 * z)
  end function yield

  !***************************************************************************!

  subroutine grad(c1, c4, z, r, gz, gr)
    ! ----------------------------------------------------------------------- !
    ! Evaluate yield function gradient
    ! ----------------------------------------------------------------------- !
    implicit none
    real(kind=rk), intent(in) :: c1, c4, z, r
    real(kind=rk), intent(out) :: gz, gr
    real(kind=rk) :: dum
    dum = c1; dum = z; dum = r
    ! dg/dz
    gz = c4
    ! dg/dr
    gr = one
  end subroutine grad

  !***************************************************************************!

  subroutine rendulic(a, az, ar, er)
    ! ----------------------------------------------------------------------- !
    ! Calculate components of a in rendulic plane
    !
    ! In
    ! --
    ! a : array
    !   Symmetric second order tensor
    !
    ! Out
    ! ---
    ! ar : float
    !   Magnitude of deviatoric part of a
    ! az : float
    !   Magnitude of isotropic part of a
    ! er : array
    !   Direction of deviatoric part of a
    ! ----------------------------------------------------------------------- !
    implicit none
    ! ...................................................................passed
    real(kind=rk), intent(in) :: a(6)
    real(kind=rk), intent(out) :: ar, az
    real(kind=rk), intent(out), optional :: er(6)
    ! ....................................................................local
    real(kind=rk) :: d(6)
    ! ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ rendulic
    az = toor3 * trace(a)
    d = dev(a)
    ar = max(epsilon(d), mag(d))
    if (present(er)) then
       if (ar > epsilon(d)) then
          er = d / ar
       else
          er = zero
       end if
    end if
    return
  end subroutine rendulic

  !***************************************************************************!

  subroutine pstiff(nx, dt, ui, d, stress, xtra, J)
    ! ----------------------------------------------------------------------- !
    ! Numerically compute material Jacobian by a centered difference scheme.
    ! ----------------------------------------------------------------------- !
    !......................................................................passed
    implicit none
    integer, intent(in) :: nx
    real(kind=rk), intent(in) :: dt, ui(*), d(6), xtra(nx), stress(6)
    real(kind=rk), intent(out) :: J(6, 6)
    !.......................................................................local
    integer :: n
    real(kind=rk) :: sigp(6), sigm(6), dp(6), dm(6), svp(nx), svm(nx), deps
    !~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ pstiff
    deps = sqrt(epsilon(d))
    do n = 1, 6
       dp = d
       dp(n) = d(n) + (deps / dt) / two
       sigp = stress
       svp = xtra
       call pcalc(nx, dt, ui, dp, sigp, svp)

       dm = d
       dm(n) = d(n) - (deps / dt) / two
       sigm = stress
       svm = xtra
       call pcalc(nx, dt, ui, dm, sigm, svm)

       J(:, n) = (sigp - sigm) / deps
    end do
  end subroutine pstiff

  !***************************************************************************!

  function mag(a)
    ! ----------------------------------------------------------------------- !
    ! Magnitude of second-order tensor
    ! ----------------------------------------------------------------------- !
    implicit none
    real(kind=rk) :: mag
    real(kind=rk), intent(in) :: a(6)
    mag = sqrt(ddp(a, a))
    return
  end function mag

  !***************************************************************************!

  function iso(a)
    ! ----------------------------------------------------------------------- !
    ! Isotropic part of second order tensor
    ! ----------------------------------------------------------------------- !
    implicit none
    real(kind=rk) :: iso(6)
    real(kind=rk), intent(in) :: a(6)
    iso = trace(a) * delta / three
    return
  end function iso

  !***************************************************************************!

  function trace(a)
    ! ----------------------------------------------------------------------- !
    ! Trace of second order tensor
    ! ----------------------------------------------------------------------- !
    implicit none
    real(kind=rk) :: trace
    real(kind=rk), intent(in) :: a(6)
    trace = sum(a(1:3))
    return
  end function trace

  !***************************************************************************!

  function dev(a)
    ! ----------------------------------------------------------------------- !
    ! Deviatoric part of second order tensor
    ! ----------------------------------------------------------------------- !
    implicit none
    real(kind=rk) :: dev(6)
    real(kind=rk), intent(in) :: a(6)
    dev = a - iso(a)
    return
  end function dev

  !***************************************************************************!

  function ddp(a, b)
    ! ----------------------------------------------------------------------- !
    ! Double dot product of second order tensors a and b
    ! ----------------------------------------------------------------------- !
    implicit none
    real(kind=rk) :: ddp
    real(kind=rk), intent(in) :: a(6), b(6)
    real(kind=rk), parameter :: w(6) = (/one, one, one, two, two, two/)
    ddp = sum(a * b * w)
    return
  end function ddp

  !***************************************************************************!

  SUBROUTINE LOGMES(MSG)
    CHARACTER*(*) MSG
    PRINT*,"INFO: "//MSG
    RETURN
  END SUBROUTINE LOGMES

  !***************************************************************************!

  SUBROUTINE BOMBED(MSG)
    CHARACTER*(*) MSG
    PRINT*,"ERROR: "//MSG//" reported from [MIG]"
    STOP
  END SUBROUTINE BOMBED

  !***************************************************************************!

  SUBROUTINE FATERR(CALLER, MSG)
    CHARACTER*(*) CALLER,MSG
    PRINT*,"FATAL ERROR: "//MSG//" reported by ["//CALLER//"]"
    STOP
  END SUBROUTINE FATERR

  !***************************************************************************!

  SUBROUTINE TOKENS(N,SA,CA)
    !    This routine converts the array of strings SA to a single character
    !    stream CA with a pipe (|) separating entries.  For example, suppose
    !
    !              SA(  1) = 'first string             '
    !              SA(  2) = 'a witty saying           '
    !              SA(  3) = '                         '
    !              SA(  4) = 'last                     '
    !
    !     Then the output of this routine is
    !
    !             CA = first string|a witty saying||last|
    !
    ! input
    ! -----
    !   N: number of strings in SA (i.e., the dimension of SA)
    !   SA: array of strings
    !
    ! output
    ! ------
    !   CA: single character stream of the strings in SA separated by pipes.
    !         BEWARE: it is the responsibility of the calling routine to
    !         dimension CA at least as large as N*(1+LEN(SA)).
    !
    !     written: 04/20/95
    !     author:  Rebecca Brannon
    !
    ! calling arguments:
    INTEGER N
    CHARACTER*(*) SA(N)
    CHARACTER*(*) CA(N)
    !      CHARACTER*1   CA(*)
    ! local:
    CHARACTER*1 PIPE,BLANK
    PARAMETER (PIPE='|',BLANK=' ')
    INTEGER I,KNT,NCHR,ICHR
    KNT=0
    DO 502 I=1,N
       DO 500 NCHR=LEN(SA(I)),1,-1
  500  IF(SA(I)(NCHR:NCHR).NE.BLANK) GO TO 7
    7  DO 501 ICHR=1,NCHR
          KNT=KNT+1
          CA(KNT)=SA(I)(ICHR:ICHR)
  501  CONTINUE
       KNT=KNT+1
       CA(KNT)=PIPE
  502 CONTINUE

    RETURN
  END SUBROUTINE TOKENS

end module plastic
