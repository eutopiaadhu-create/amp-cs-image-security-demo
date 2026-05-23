"""
Permission Management for multi-level access control.
Implements three-tier permission system: Guest, UserA, UserB (Admin).
"""

from enum import Enum
from typing import Optional
from dataclasses import dataclass


class UserLevel(Enum):
    """User permission levels."""
    GUEST = 0       # Can only view steganographic image
    USER_A = 1      # Can decrypt target images (partial access)
    USER_B = 2      # Can decrypt source + target images (full access)
    ADMIN = 3       # Same as USER_B + system management


@dataclass
class User:
    """User information."""
    username: str
    level: UserLevel
    display_name: str = ""
    
    def __post_init__(self):
        if not self.display_name:
            self.display_name = self.username


class PermissionManager:
    """
    Manages user authentication and permission levels.
    """
    
    USERS = {
        'a': {'password': '123', 'level': UserLevel.USER_A, 'display': 'A用户'},
        'b': {'password': '456', 'level': UserLevel.USER_B, 'display': 'B用户'},
        'admin': {'password': 'root', 'level': UserLevel.ADMIN, 'display': '管理员'},
    }
    
    def __init__(self):
        self.current_user: Optional[User] = None
    
    def authenticate(self, username: str, password: str) -> Optional[User]:
        """
        Authenticate user with username and password.
        
        Returns:
            User object if authenticated, None otherwise
        """
        username = username.lower().strip()
        
        if username not in self.USERS:
            return None
        
        user_info = self.USERS[username]
        if user_info['password'] != password:
            return None
        
        self.current_user = User(
            username=username,
            level=user_info['level'],
            display_name=user_info['display']
        )
        return self.current_user
    
    def guest_login(self) -> User:
        """Login as guest user."""
        self.current_user = User(
            username='guest',
            level=UserLevel.GUEST,
            display_name='访客'
        )
        return self.current_user
    
    def logout(self):
        """Logout current user."""
        self.current_user = None
    
    @property
    def is_logged_in(self) -> bool:
        return self.current_user is not None
    
    @property
    def level(self) -> UserLevel:
        if self.current_user is None:
            return UserLevel.GUEST
        return self.current_user.level
    
    def can_view_stego(self) -> bool:
        """All users can view steganographic image."""
        return True
    
    def can_decrypt_target(self) -> bool:
        """UserA and above can decrypt target images."""
        return self.level.value >= UserLevel.USER_A.value
    
    def can_decrypt_source(self) -> bool:
        """Only UserB and Admin can decrypt source images."""
        return self.level.value >= UserLevel.USER_B.value
    
    def should_apply_privacy_protection(self) -> bool:
        """
        Determine if privacy protection should be applied.
        Lower permission users see protected (blurred/mosaic) sensitive regions.
        """
        return self.level.value < UserLevel.USER_B.value
    
    def get_accessible_tabs(self) -> list:
        """Get list of UI tabs accessible to current user."""
        tabs = ['访客视图']
        
        if self.can_decrypt_target():
            tabs.append('A用户 (部分访问)')
        
        if self.can_decrypt_source():
            tabs.append('B用户 (完全访问)')
        
        return tabs
