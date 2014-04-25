subroutine plastic_check(ui)
  use plastic, only: pchk
  implicit none
  integer, parameter :: rk=selected_real_kind(14)
  integer, parameter :: nui=5
  real(kind=rk), intent(inout) :: ui(nui)
  call pchk(ui)
  return
end subroutine plastic_check
subroutine plastic_request_xtra(ui, nxtra, namea, keya, xtra)
  use plastic, only: prxv
  implicit none
  integer, parameter :: rk=selected_real_kind(14)
  integer, parameter :: nui=5, nx=2
  real(kind=rk), intent(in) :: ui(nui)
  integer, intent(out) :: nxtra
  character(1), intent(out) :: namea(nx*50), keya(nx*10)
  real(kind=rk), intent(out) :: xtra(nx)
  call prxv(ui, nxtra, namea, keya, xtra)
  return
end subroutine plastic_request_xtra
subroutine plastic_update_state(nx, dt, ui, d, stress, xtra)
  use plastic, only: pcalc
  integer, parameter :: rk=selected_real_kind(14)
  integer, parameter :: nui=5
  integer, intent(in) :: nx
  real(kind=rk), intent(in) :: dt, ui(nui), d(6)
  real(kind=rk), intent(inout) :: stress(6), xtra(nx)
  call pcalc(nx, dt, ui, d, stress, xtra)
  return
end subroutine plastic_update_state
subroutine plastic_stiff(nx, dt, ui, d, stress, xtra, J)
  use plastic, only: pstiff
  integer, parameter :: rk=selected_real_kind(14)
  integer, parameter :: nui=5
  integer, intent(in) :: nx
  real(kind=rk), intent(in) :: dt, ui(nui), d(6)
  real(kind=rk), intent(in) :: stress(6), xtra(nx)
  real(kind=rk), intent(out) :: J(6, 6)
  call pstiff(nx, dt, ui, d, stress, xtra, J)
  return
end subroutine plastic_stiff
